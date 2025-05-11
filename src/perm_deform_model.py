import torch
import torch.nn.functional as F

from torch import nn
from utils.general_utils import get_embedder


class PermDeformModel(nn.Module):
    def __init__(self, perm, device):
        super().__init__()

        self.perm = perm
        self.device = device
        
        self.pts_freq = 8
        self.pts_embedder, self.embed_size = get_embedder(
            10,
            3,
            use_gauss_encoding=False,
            gauss_sigma=1.0,
        )
        self.init_networks()

    def init_networks(self):       
        self.deformNet = MLP(
            input_dim=self.embed_size+136,
            output_dim=9,
            hidden_dim=256,
            hidden_layers=6
        )

    def training_setup(self):
        params_group = [
            {'params': self.deformNet.parameters(), 'lr': 1e-4},
        ]
        self.optimizer = torch.optim.Adam(params_group, betas=(0.9, 0.999))

    def compute_mlp_delta_coef(self, vert_embed, condition, mlp):
        uv_vertices_shape_embeded_condition = torch.cat((vert_embed, condition), dim=-1)[None]
        deforms = mlp(uv_vertices_shape_embeded_condition)
        deforms = torch.tanh(deforms)
        uv_vertices_deforms = deforms[..., :3]
        rot_delta_0 = deforms[..., 3:7]
        rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)
        rot_delta_v = rot_delta_0[..., 1:]
        rot_delta = torch.cat((rot_delta_r, rot_delta_v), dim=-1)
        scale_coef = deforms[..., 7:]
        scale_coef = torch.exp(scale_coef)

        return uv_vertices_deforms[0], rot_delta[0], scale_coef[0]
    
    def decode(self, gaussians, codedict):
        roots = codedict['roots']
        theta = codedict['theta']
        beta = codedict['beta']
        perm_out = self.perm(roots=roots, theta=theta, beta=beta)

        strands = perm_out["strands"].position[0]
        strands = strands[:, ::2]
        N, C, _ = strands.shape
        strands = strands.reshape(N * C, -1)
        strands = gaussians.perm2scene(strands)
        strands_enc = self.pts_embedder(strands)

        coef = perm_out["coef"][0].unsqueeze(1).expand(-1, C, -1).reshape(N * C, -1)
        rotation = codedict['R'].flatten()[None].expand(N * C, -1)
        translation = self.pts_embedder(codedict['T'][None].expand(N * C, -1))
        condition = torch.cat((coef, rotation, translation), dim=-1)

        # hair_deforms, strand_rot_delta, strand_scale_coef = self.compute_mlp_delta_coef(
        #     strands_enc,
        #     condition,
        #     self.deformNet
        # )
        strands_final = strands

        guide_strands = perm_out["guide_strands"].position[0]
        guide_strands = guide_strands[:, ::2]
        guide_strands = guide_strands.reshape(-1, 3)
        guide_strands = gaussians.perm2scene(guide_strands)

        return strands_final, guide_strands, None, None
    
    def capture(self):
        return (
            self.deformNet.state_dict(),
            self.optimizer.state_dict(),
        )
    
    def restore(self, model_args):
        (net_dict,
         opt_dict) = model_args
        self.deformNet.load_state_dict(net_dict)
        self.training_setup()
        self.optimizer.load_state_dict(opt_dict)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, hidden_layers=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fcs = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for i in range(hidden_layers-1)]
        )
        self.output_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        # input: B,V,d
        batch_size, N_v, input_dim = input.shape
        input_ori = input.reshape(batch_size*N_v, -1)
        h = input_ori
        for i, l in enumerate(self.fcs):
            h = self.fcs[i](h)
            h = F.relu(h)
        output = self.output_linear(h)
        output = output.reshape(batch_size, N_v, -1)

        return output