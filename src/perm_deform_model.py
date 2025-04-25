import sys
import pickle
import torch
import numpy as np
from torch import nn
import math
import torch.nn.functional as F
from pytorch3d.io import load_obj

from deform_model import DeformModel
from flame import FLAME_mica, parse_args
from utils.general_utils import Pytorch3dRasterizer, Embedder, load_binary_pickle, a_in_b_torch, face_vertices_gen


class PermDeformModel:
    def __init__(self, perm, device):
        self.perm = perm
        self.device = device
        
        self.pts_freq = 8
        self.pts_embedder = Embedder(self.pts_freq)
        self.init_networks()

    def init_networks(self):       
        self.deformNet = MLP(
            input_dim=self.pts_embedder.dim_embeded+78,
            output_dim=10,
            hidden_dim=256,
            hidden_layers=6
        )

    def training_setup(self):
        params_group = [
            {'params': self.deformNet.parameters(), 'lr': 1e-4},
        ]
        self.optimizer = torch.optim.Adam(params_group, betas=(0.9, 0.999))

    def compute_mlp_delta_coef(self, vert_embed, condition, mlp):
        uv_vertices_shape_embeded_condition = torch.cat((vert_embed, condition), dim=2)
        deforms = mlp(uv_vertices_shape_embeded_condition)
        deforms = torch.tanh(deforms)
        uv_vertices_deforms = deforms[..., :3]
        rot_delta_0 = deforms[..., 3:7]
        rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)
        rot_delta_v = rot_delta_0[..., 1:]
        rot_delta = torch.cat((rot_delta_r, rot_delta_v), dim=-1)
        scale_coef = deforms[..., 7:]
        scale_coef = torch.exp(scale_coef)

        return uv_vertices_deforms, rot_delta, scale_coef
    
    def decode(self, roots, codedict):
        theta = codedict['theta']
        beta = codedict['beta']
        perm_out = self.perm(roots=roots, theta=theta, beta=beta)

        coef = perm_out["coef"]
        rotation = codedict['R'].flatten().detach()
        translation = codedict['T'].flatten().detach()
        condition = torch.cat((coef, rotation, translation))

        hair_deforms, strand_rot_delta, strand_scale_coef = self.compute_mlp_delta_coef(
            self.pts_embedder(roots),
            condition,
            self.hairDeformNet
        )
        strands_final = perm_out["strands"] + hair_deforms

        return strands_final, strand_rot_delta, strand_scale_coef
    
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