import torch
import trimesh
import pickle
import numpy as np
import torch.nn as nn

from utils.general_utils import inverse_sigmoid, get_expon_lr_func, strip_symmetric, build_scaling_rotation, quatProduct_batch

STRAND_VERTEX_COUNT = 50

class GaussianPerm(nn.Module):
    def __init__(self, perm, pseudo_roots, start_hair_style, sh_degree, num_strands=20000):
        super(GaussianPerm, self).__init__()

        self.optimizer = None

        self.max_sh_degree = sh_degree
        self.num_strands = num_strands

        self.perm = perm
        self.roots = self.perm.hair_roots.sample_scalp_mesh(num_strands, pseudo_roots)

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
        self.segmentation_activation = torch.sigmoid


        self.init_parameters(start_hair_style)

    def init_parameters(self, start_hair_style):
        # --- initialize perm parameters (theta, beta) ---
        if start_hair_style is not None:
            self.theta = nn.Parameter(torch.tensor(start_hair_style["theta"])[None]
                                    .requires_grad_(True))
            self.beta  = nn.Parameter(torch.tensor(start_hair_style["beta"])[None]
                                    .requires_grad_(True))
            # self.beta  = nn.Parameter(self.perm.beta_avg().requires_grad_(True))
        else:
            self.theta = nn.Parameter(self.perm.theta_avg().requires_grad_(True))
            self.beta  = nn.Parameter(self.perm.beta_avg().requires_grad_(True))

        full_strands = self.perm(
            roots=self.get_roots_xyz[None].cuda(),
            theta=self.theta.cuda(),
            beta=self.beta.cuda()
        )["strands"].position[0]
        base_strands = full_strands[:, ::2, :] / 100.0

        # import pdb; pdb.set_trace()

        S, N_ds, _ = base_strands.shape
        gaussian_count = S * (N_ds - 1)
        self.active_sh_degree = 0

        # --- compute segment directions and lengths ---
        # edges: (S, N_ds-1, 3)
        edges = base_strands[:, 1:, :] - base_strands[:, :-1, :]
        # flatten to (M, 3)
        edge_dirs = edges.reshape(-1, 3)
        # lengths: (M, 1)
        edge_lens = edge_dirs.norm(dim=1, keepdim=True)
        M = edge_dirs.shape[0]
        device = edge_dirs.device
        eps = 1e-6

        # --- initialize scales: (M, 2) ---
        # y-scale = segment length, x/z-scale = length / 5
        scale_y  = edge_lens[:, 0]
        scale_xz = scale_y / 10000.0
        raw_scales = torch.stack([scale_xz, scale_y], dim=1).to(device)
        # invert the activation (log) so that exp(_scaling_base) == desired raw_scales
        base_scales = self.scaling_inverse_activation(raw_scales)
        self._scaling_base = nn.Parameter(base_scales.requires_grad_(True))
        self._scaling = None

        # --- initialize spherical-harmonic features: (M,3,n_sh) ---
        avg_color = torch.tensor([0.12, 0.13, 0.14], device=device) * 0.0
        n_sh = (self.max_sh_degree + 1) ** 2
        feats = torch.zeros((M, 3, n_sh), device=device)
        feats[:, :, 0] = avg_color.unsqueeze(0)

        # split into DC and rest
        dc = feats[:, :, 0:1].transpose(1, 2).contiguous()   # (M,1,3)
        rest = feats[:, :, 1:].transpose(1, 2).contiguous()  # (M,n_sh-1,3)
        self._features_dc   = nn.Parameter(dc.requires_grad_(True))
        self._features_rest = nn.Parameter(rest.requires_grad_(True))

        # --- initialize opacity: (M,1) ---
        opacs = inverse_sigmoid(
            1.0 * torch.ones((M, 1), dtype=torch.float, device=device)
        )
        self._opacity = nn.Parameter(opacs.requires_grad_(True))
        
        # self._seg_logit = nn.Parameter(torch.ones((M, 1), device=device))

        # --- placeholder for Gaussian centers/rotations ---
        self._xyz = None
        self._rotation = None


    def capture(self):
        return (
            self.active_sh_degree,
            self._features_dc,
            self._features_rest,
            self._scaling_base,
            self._opacity,
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
        self._opacity,
        opt_dict, 
        self.spatial_lr_scale,
        self.roots,
        self.theta,
        self.beta) = model_args
        self.training_setup(training_args)
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_roots_xyz(self):
        return self.roots

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    # @property
    # def get_seg_logit(self):
    #     return self.segmentation_activation(self._seg_logit)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def training_setup(self, training_args):
        self.spatial_lr_scale = 5

        l = [
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
                "params": [self._scaling_base],
                "lr": training_args.scaling_lr * self.spatial_lr_scale,
                "name": "scaling",
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
            if param_group["name"] == "beta" or param_group["name"] == "theta":
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

    def perm2scene(self, points):
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        points_corr = torch.stack([x, y, z], dim=1) / 100.0

        return points_corr
    
    def compute_edge_quats(self, edge_dirs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Given edge direction vectors (M,3) return unit quaternions (M,4)
        which rotate the world +Y axis to each edge direction.
        """
        up   = torch.tensor([0.0, 1.0, 0.0], device=edge_dirs.device).expand_as(edge_dirs)
        b    = edge_dirs / (edge_dirs.norm(dim=1, keepdim=True) + eps)
        dot  = (up * b).sum(dim=1, keepdim=True).clamp(-1 + eps, 1 - eps)
        axis = torch.cross(up, b, dim=1)
        axis = axis / (axis.norm(dim=1, keepdim=True) + eps)
        ang  = torch.acos(dot)
        qw   = torch.cos(ang * 0.5)
        qxyz = axis * torch.sin(ang * 0.5)
        return torch.cat([qw, qxyz], dim=1)  # (M,4)
    
    def update_xyz_rot_scale(self, strand_vertices, rot_delta, scale_coeff):
        """
        strand_vertices : (S*V,3) vertices output by deformer
        Re‑computes mid‑points and edge‑quaternions; updates xyz, scaling, rotation buffers.
        """
        pts = strand_vertices.reshape(self.num_strands, STRAND_VERTEX_COUNT, 3)
        mids = 0.5 * (pts[:, :-1] + pts[:, 1:])                  # (S,V-1,3)
        edges = pts[:, 1:, :] - pts[:, :-1, :]                   # (S,V-1,3)
        edge_dirs = edges.reshape(-1, 3)

        # update buffers (not learnable)
        self._xyz      = mids.reshape(-1, 3)
        self._rotation = self.compute_edge_quats(edge_dirs)
        # scaling keeps the same learnable bases; _scaling is cached form
        self._scaling  = torch.cat([self._scaling_base[:, [0]],
                                    self._scaling_base[:, [1]],
                                    self._scaling_base[:, [0]]],
                                   dim=-1)