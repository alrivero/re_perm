import torch
import torch.nn.functional as F

from torch import nn
from utils.general_utils import get_embedder
from flame.flame import FlameHead
from utils.misc import build_rotation_matrix

class PermDeformModel(nn.Module):
    def __init__(self, perm, og_pos_trans, smplx_joints, smplx_params, device, perm_scale=100.0):
        super().__init__()

        self.perm = perm
        self.perm_scale = perm_scale
        self.device = device

        # We need these to go from PERM to Scene
        self.og_pos_trans = torch.tensor(og_pos_trans).float().to(device)
        self.head_joint = smplx_joints[15].float().to(device)
        self.global_joint = smplx_joints[0].float().to(device)
        self.global_rot = torch.from_numpy(smplx_params["global_rot"]).float().to(device)
        self.global_trans = torch.from_numpy(smplx_params["global_trans"]).float().to(device)
        self.global_scale = torch.from_numpy(smplx_params["global_scale"]).float().to(device)
        self.head = torch.from_numpy(self.perm.hair_roots.head.vertices).float().to(self.device)
        self.head_normals = torch.from_numpy(self.perm.hair_roots.head.vertex_normals).float().to(self.device)
        
    #     self.pts_freq = 8
    #     self.pts_embedder, self.embed_size = get_embedder(
    #         10,
    #         3,
    #         use_gauss_encoding=False,
    #         gauss_sigma=1.0,
    #     )
    #     self.init_networks()

    # def init_networks(self):       
    #     self.deformNet = MLP(
    #         input_dim=self.embed_size+136,
    #         output_dim=9,
    #         hidden_dim=256,
    #         hidden_layers=6
    #     )

    # def training_setup(self):
    #     params_group = [
    #         {'params': self.deformNet.parameters(), 'lr': 1e-4},
    #     ]
    #     self.optimizer = torch.optim.Adam(params_group, betas=(0.9, 0.999))

    # def compute_mlp_delta_coef(self, vert_embed, condition, mlp):
    #     uv_vertices_shape_embeded_condition = torch.cat((vert_embed, condition), dim=-1)[None]
    #     deforms = mlp(uv_vertices_shape_embeded_condition)
    #     deforms = torch.tanh(deforms)
    #     uv_vertices_deforms = deforms[..., :3]
    #     rot_delta_0 = deforms[..., 3:7]
    #     rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)
    #     rot_delta_v = rot_delta_0[..., 1:]
    #     rot_delta = torch.cat((rot_delta_r, rot_delta _v), dim=-1)
    #     scale_coef = deforms[..., 7:]
    #     scale_coef = torch.exp(scale_coef)

    #     return uv_vertices_deforms[0], rot_delta[0], scale_coef[0]

    def obtain_global_matx(
        self,
        rot_mat: torch.Tensor,    # [3×3] rotation matrix
        trans_vec: torch.Tensor,  # [3] translation vector
        scale: float,             # uniform scale
        joint: torch.Tensor       # [3] joint‐position to rotate around
    ):
        """
        Build a global 3×3 matrix M and 3-vector v such that:
            x' = M @ x + v
        where rotation is performed about `joint`, then scaled,
        then translated by trans_vec.

        Args:
        rot_mat   (3×3) – desired rotation
        trans_vec (3,)  – desired translation
        scale      float – uniform scale factor
        joint     (3,)  – pivot point for the rotation

        Returns:
        M (3×3), v (3,)
        """
        device = rot_mat.device

        # build uniform scale matrix
        S = torch.eye(3, device=device) * scale     # [3×3]

        # first scale, then rotate (same order as original code: R @ S)
        M = rot_mat.to(device) @ S                  # [3×3]

        # compute translation term so that rotation happens about `joint`:
        #   x' = M(x – joint) + joint + trans_vec
        #  => x' = M x + (–M joint + joint + trans_vec)
        v = trans_vec.to(device) + joint.to(device) - M @ joint.to(device)

        return M, v
    
    def decode(self, gaussians, codedict):
        # Decode our strands
        roots = codedict['roots']
        theta = codedict['theta']
        beta = codedict['beta']
        perm_out = self.perm(roots=roots, theta=theta, beta=beta)

        # Now, reshape and get them out of PERM space
        strands = perm_out["strands"].position[0]
        guide_strands = perm_out["guide_strands"].position[0]

        N, C, _ = strands.shape
        strands = strands.reshape(N * C, -1)
        guide_strands = guide_strands.reshape(-1, 3)
        scalp = gaussians.scalp_roots
        head = self.head
        head_normals = self.head_normals

        strands_can       = strands / self.perm_scale
        guide_strands_can = guide_strands / self.perm_scale
        scalp_can         = scalp / self.perm_scale
        head_can          = head / self.perm_scale
        normals_can       = head_normals

        strands_def = strands_can + self.og_pos_trans
        guide_strands_def = guide_strands_can + self.og_pos_trans
        scalp_def = scalp_can + self.og_pos_trans
        head_def = head_can + self.og_pos_trans

        # Apply initial global transformation
        R = build_rotation_matrix(self.global_rot.to(self.device))  # [3×3]
        M = R * self.global_scale
        joint = self.global_joint.to(self.device)
        tvec  = self.global_trans.to(self.device)
        v = tvec + joint - M @ joint  # [3]

        strands_def       = strands_def       @ M.T + v
        guide_strands_def = guide_strands_def @ M.T + v
        scalp_def         = scalp_def         @ M.T + v
        head_def          = head_def          @ M.T + v
        normals_def       = F.normalize(normals_can @ R.T, dim=1)

        # Apply learned transformation
        R_delta = codedict['learned_rot'].to(self.device)   # [3×3]
        t_delta = codedict['learned_trans'].to(self.device) # [3]

        strands_final       = strands_def       @ R_delta.T + t_delta
        guide_strands_final = guide_strands_def @ R_delta.T + t_delta
        scalp_final         = scalp_def         @ R_delta.T + t_delta
        head_final          = head_def          @ R_delta.T + t_delta
        normals_final       = F.normalize(normals_def @ R_delta.T, dim=1)

        perm_coeff = perm_out["coef"][0]
        return strands_can, guide_strands_can, strands_final, guide_strands_final, scalp_can, scalp_final, head_can, head_final, normals_can, normals_final, perm_coeff
    
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