import torch
import numpy as np
import torch.nn as nn

from utils.general_utils import inverse_sigmoid, get_expon_lr_func, strip_symmetric, build_scaling_rotation

STRAND_VERTEX_COUNT = 100

class GaussianPerm(nn.Module):
    def __init__(self, perm, sh_degree, num_strands=10000):
        super(GaussianPerm, self).__init__()

        self.optimizer = None

        self.max_sh_degree = sh_degree
        self.num_strands = num_strands

        self.perm = perm
        self.scalp_mask = np.load(
            "/home/alrivero/ENPC/sparkly_perm/src/assets/FLAME_masks/FLAME_masks.pkl")["scalp"]

        self.init_parameters()

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def init_parameters(self):
        gaussian_count = self.num_strands * STRAND_VERTEX_COUNT

        # Save PERM and also the items we'll need to optimize
        sampled_roots = self.perm.hair_roots.sample_scalp_surface(self.scalp_mask, self.num_strands)
        self.roots = nn.Parameter(sampled_roots.requires_grad_(True))
        self.theta = nn.Parameter(self.perm.theta_avg().requires_grad_(True))
        self.beta = nn.Parameter(self.perm.beta_avg().requires_grad_(True))
        
        # Save Gaussian Paramters
        self.active_sh_degree = 0

        features = torch.rand((gaussian_count, 3, (self.max_sh_degree + 1) ** 2)).float()
        features[:, 3:, 1:] = 0.0

        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )

        self._scaling = nn.Parameter(torch.rand((gaussian_count, 3)).requires_grad_(True))

        rots = torch.zeros((gaussian_count, 4))
        rots[:, 0] = 1
        self._rotation = nn.Parameter(rots.requires_grad_(True))

        opacities = inverse_sigmoid(
            0.1
            * torch.ones(
                (gaussian_count, 1), dtype=torch.float
            )
        )
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        # Not actually learned, but still integral
        self.max_radii2D = torch.zeros((gaussian_count))
        self.xyz_gradient_accum = torch.zeros((gaussian_count, 1), device="cuda")
        self.denom = torch.zeros((gaussian_count, 1), device="cuda")

    def capture(self):
        return (
            self.active_sh_degree,
            self._features_dc,
            self._features_rest,
            self._scaling_base,
            self._rotation_base,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.roots,
            self.theta,
            self.beta
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._features_dc, 
        self._features_rest,
        self._scaling_base, 
        self._rotation_base, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self.roots,
        self.theta,
        self.beta) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        # REWRITE ME
        return None

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.spatial_lr_scale = 5

        l = [
            {
                "params": [self.roots],
                "lr": training_args.perm_lr_init,
                "name": "roots",
            },
            {
                "params": [self.theta],
                "lr": training_args.perm_lr_init,
                "name": "theta",
            },
            {
                "params": [self.beta],
                "lr": training_args.perm_lr_init,
                "name": "beta",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.perm_lr_init * self.spatial_lr_scale,
            lr_final=training_args.perm_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.perm_lr_delay_mult,
            max_steps=training_args.perm_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "perm":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
    
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]