import math
import os
import torch
import torch.nn as nn

from utils.spec_utils import RenderingEquationEncoding, positional_encoding
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_linear_noise_func

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

class UncertaintyRender(nn.Module):
    def __init__(
        self,
        viewpe: int = 2,
        featureC: int = 128,
        num_theta: int = 4,
        num_phi: int = 8,
        init_alpha: float = 0.7366779322422883,
        init_beta:  float = 0.830864306541151,
    ):
        super().__init__()
        self.num_theta = num_theta
        self.num_phi   = num_phi
        self.viewpe    = viewpe

        # Save initial Beta guesses
        self.init_alpha = init_alpha
        self.init_beta  = init_beta

        # Total input channels (unchanged)
        self.in_mlpC = (
            3 * viewpe * 2
            + 2 * viewpe * 3
            + 3
            + num_theta * num_phi * 2
            + 1
        )

        self.ree_function = RenderingEquationEncoding(
            num_theta, num_phi, device='cuda'
        )

        # MLP now outputs only 2 values: raw alpha, raw beta
        layer1 = nn.Linear(self.in_mlpC, featureC)
        layer2 = nn.Linear(featureC, featureC)
        layer3 = nn.Linear(featureC, 2)

        self.mlp = nn.Sequential(
            layer1,
            nn.ReLU(inplace=True),
            layer2,
            nn.ReLU(inplace=True),
            layer3,
        )

        # Initialize near zero
        def _init_near_zero(m, std=1e-6):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                nn.init.normal_(m.bias,   mean=0.0, std=std)
        self.mlp.apply(lambda m: _init_near_zero(m))

        # Set layer3 bias so softplus(raw_alpha)=init_alpha etc.
        with torch.no_grad():
            raw_alpha_bias = math.log(math.expm1(self.init_alpha))
            raw_beta_bias  = math.log(math.expm1(self.init_beta))
            layer3.bias.data.copy_(
                torch.tensor([raw_alpha_bias, raw_beta_bias],
                             device=layer3.bias.device)
            )

    def reflect(self, viewdir, normal):
        return 2 * (viewdir * normal).sum(dim=-1, keepdim=True) * normal - viewdir

    def safe_normalize(self, x, eps=1e-8):
        return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

    @staticmethod
    def _beta_grep_sample(alpha: torch.Tensor,
                          beta:  torch.Tensor,
                          *,
                          eps_min: float = 1e-6):
        """
        Pathwise sample z ~ Beta(α,β) using the G-REP logistic transform,
        and return the two g_corr terms needed for unbiased gradients.

        Returns
        -------
        z         : torch.Tensor   -- sample in (0,1)
        g_corr_a  : torch.Tensor   -- correction term for α
        g_corr_b  : torch.Tensor   -- correction term for β
        """
        # Logistic base noise ε
        u   = torch.rand_like(alpha).clamp_(eps_min, 1 - eps_min)
        eps = torch.log(u) - torch.log1p(-u)                     # ε ~ Logistic(0,1)

        # Scale σ(α,β)  = √(ψ₁(α) + ψ₁(β))
        sigma = torch.sqrt(torch.polygamma(1, alpha) + torch.polygamma(1, beta))

        # Pathwise transform (Eq. 38 in G-REP paper)
        z = torch.sigmoid(eps * sigma + torch.digamma(alpha) - torch.digamma(beta))

        # g_corr terms (Suppl. Eqs. 43–44)
        log_z   = torch.log(z.clamp_(eps_min, 1 - eps_min))
        log1mz  = torch.log1p(-z).clamp_(min=-15.)
        psi_sum = torch.digamma(alpha + beta)

        g_corr_a = log_z  - torch.digamma(alpha) + psi_sum
        g_corr_b = log1mz - torch.digamma(beta)  + psi_sum

        return z, g_corr_a, g_corr_b
    # ──────────────────────────────────────────────────────────────────

    def forward(self, viewdirs, features, normal, uni_logit, pos, frame_ids):
        N = viewdirs.shape[0]

        # --- ASG → view-dependent colour features (unchanged) ----------
        asg = features.view(N, self.num_theta, self.num_phi, 4)
        a, la, mu = torch.split(asg, [2, 1, 1], dim=-1)

        refl = self.reflect(-viewdirs, normal)
        rd   = self.safe_normalize(refl)
        cf   = self.ree_function(rd, a, la, mu).view(N, -1)

        ndv = ((-viewdirs) * normal).sum(-1, keepdim=True)
        parts = [cf, ndv]
        if self.viewpe > -1:
            parts.append(viewdirs)
        if self.viewpe > 0:
            parts.append(positional_encoding(viewdirs, self.viewpe))
            aux = torch.cat(
                [pos.unsqueeze(-1), frame_ids.unsqueeze(-1), uni_logit], dim=-1
            )
            parts.append(positional_encoding(aux, self.viewpe))

        x = torch.cat(parts, dim=-1)

        # --- Predict α, β ------------------------------------------------
        raw_ab          = self.mlp(x)              # (N, 2)
        a_raw, b_raw    = raw_ab.chunk(2, dim=1)
        alpha           = F.softplus(a_raw) + 1e-6
        beta            = F.softplus(b_raw) + 1e-6

        # --- G-REP sample + correction ----------------------------------
        z_beta, gca, gcb = self._beta_grep_sample(alpha, beta)
        # expose for an outer loss hook (optional)
        self._alpha, self._beta = alpha, beta
        self._g_corr_a, self._g_corr_b = gca, gcb

        # --- Relaxed spike ------------------------------------------------
        pi_q = torch.sigmoid(uni_logit)                           # (N,1)
        r    = RelaxedBernoulli(temperature=0.1, probs=pi_q).rsample()

        # --- Mixture ------------------------------------------------------
        g = r + (1.0 - r) * z_beta                                # (N,1)
        return g

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
        return u

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