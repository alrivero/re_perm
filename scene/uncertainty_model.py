import math
import os
import torch
import torch.nn as nn

from utils.spec_utils import RenderingEquationEncoding, positional_encoding
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_linear_noise_func

class UncertaintyRender(nn.Module):
    def __init__(
        self,
        viewpe: int = 2,
        featureC: int = 128,
        num_theta: int = 4,
        num_phi: int = 8,
    ):
        super().__init__()

        self.num_theta = num_theta
        self.num_phi   = num_phi
        self.ch_normal_dot_viewdir = 1
        self.viewpe = viewpe

        # total input channels
        self.in_mlpC = (
            3 * viewpe * 2 
            + 2 * viewpe * 3
            + 3
            + num_theta * num_phi * 2
            + self.ch_normal_dot_viewdir
        )

        self.ree_function = RenderingEquationEncoding(
            num_theta, num_phi, device='cuda'
        )

        # build MLP
        layer1 = nn.Linear(self.in_mlpC, featureC)
        layer2 = nn.Linear(featureC, featureC)
        layer3 = nn.Linear(featureC, 1)  # raw logit output

        self.mlp = nn.Sequential(
            layer1,
            nn.ReLU(inplace=True),
            layer2,
            nn.ReLU(inplace=True),
            layer3,
        )

        def _init_near_zero(m, std=1e-6):
            if isinstance(m, nn.Linear):
                # break symmetry with a small gaussian noise
                nn.init.normal_(m.weight, mean=0.0, std=std)
                # bias near zero
                nn.init.normal_(m.bias,   mean=0.0, std=std)

        self.mlp.apply(lambda m: _init_near_zero(m, std=1e-6))

    def reflect(self, viewdir, normal):
        return 2 * (viewdir * normal).sum(dim=-1, keepdim=True) * normal - viewdir

    def safe_normalize(self, x, eps=1e-8):
        return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))

    def forward(self, viewdirs, features, normal, uni_logit, pos, frame_ids):
        # unpack ASG params
        N = viewdirs.shape[0]
        asg = features.view(N, self.num_theta, self.num_phi, 4)
        a, la, mu = torch.split(asg, [2,1,1], dim=-1)

        # rendering eq. features
        refl = self.reflect(-viewdirs, normal)
        rd   = self.safe_normalize(refl)
        cf   = self.ree_function(rd, a, la, mu)
        cf   = cf.view(N, -1)

        # build MLP input
        ndv = ((-viewdirs) * normal).sum(dim=-1, keepdim=True)
        parts = [cf, ndv]
        if self.viewpe > -1:
            parts.append(viewdirs)
        if self.viewpe > 0:
            parts.append(positional_encoding(viewdirs, self.viewpe))
            aux = torch.cat([pos.unsqueeze(-1), frame_ids.unsqueeze(-1), uni_logit], dim=-1)
            parts.append(positional_encoding(aux, self.viewpe))

        x = torch.cat(parts, dim=-1)
        
        # reparameterization trick: uni_logit is μ, mlp(x) predicts log σ
        log_sigma = self.mlp(x)            
        sigma     = torch.exp(log_sigma)   
        eps       = torch.randn_like(sigma)
        u         = uni_logit + sigma * eps
        return u

class UncertaintyNetwork(nn.Module):
    def __init__(self):
        super(UncertaintyNetwork, self).__init__()

        self.asg_feature = 24
        self.num_theta = 4
        self.num_phi = 8
        self.view_pe = 2
        self.hidden_feature = 128
        self.asg_hidden = self.num_theta * self.num_phi * 4

        self.gaussian_feature = nn.Linear(self.asg_feature, self.asg_hidden)

        self.render_module = UncertaintyRender(self.view_pe, self.hidden_feature, self.num_theta, self.num_phi)

    def forward(self, lobes, view, normal, pos, frame_id):
        uni_logit = lobes[:, [0]]
        lobes = lobes[:, 1:]

        feature = self.gaussian_feature(lobes)
        frame_ids = torch.zeros_like(frame_id[None].repeat(len(lobes)))

        u = self.render_module(view, feature, normal, uni_logit, pos, frame_ids)
        return torch.sigmoid(u)

class UncertaintyModel():
    def __init__(self):
            self.uncertainty = UncertaintyNetwork().cuda()
            self.optimizer = None
            self.spatial_lr_scale = 5

    def step(self, asg_feature, viewdir, normal, pos, lobes):
        return self.uncertainty(asg_feature, viewdir, normal, pos, lobes)

    def train_setting(self, training_args):
        l = [
            {'params': list(self.uncertainty.parameters()),
            'lr': training_args.gate_lr / 5,
            "name": "uncertainty"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.uncertainty_scheduler_args = get_linear_noise_func(lr_init=training_args.feature_lr,
                                                            lr_final=training_args.feature_lr / 20,
                                                            lr_delay_mult=training_args.position_lr_delay_mult,
                                                            max_steps=training_args.uncertainty_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "uncertainty/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.uncertainty.state_dict(), os.path.join(out_weights_path, 'uncertainty.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "uncertainty"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "uncertainty/iteration_{}/uncertainty.pth".format(loaded_iter))
        self.uncertainty.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "uncertainty":
                lr = self.uncertainty_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr