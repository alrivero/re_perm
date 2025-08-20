import torch
import trimesh
import pickle
import math
import numpy as np
import torch.nn as nn
import scipy.spatial as _sp
import scipy
from pytorch3d.ops import knn_points


from random import random
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, strip_symmetric, build_scaling_rotation, quatProduct_batch
# SCALE SCALE SCALE

STRAND_VERTEX_COUNT = 100
SCALE_DIVISOR = 1

class GaussianPerm(nn.Module):
    def __init__(self, perm,
                 start_hair_style, sh_degree, asg_degree, global_scale,
                 num_strands=2500,
                 neighbor_k=6,
                 num_gaussians=2000000):
        super().__init__()

        self.perm          = perm
        self.num_strands   = num_strands
        self.max_sh_degree = sh_degree
        self.max_asg_degree = asg_degree
        self.neighbor_k    = neighbor_k
        self.uniform_strand_color = False
        self.global_scale = global_scale

        # sample scalp roots once
        self.roots_unculled, self.global_strand_radii, self.roots = self.perm.hair_roots.sample_scalp_hex(
            num_strands
        )

        self.roots = self.roots.to(self.perm.device)
        self.scalp_roots = self.roots.clone()
        self.scalp_normals = self.compute_root_normals()  # uses self.roots internally

        # offset distance (meters) to avoid Z-fighting for scalp disks
        self.scalp_offset = 0.0

        self.num_strands = len(self.roots)

        # activations / helpers
        self.scaling_activation         = torch.exp
        self.scaling_inverse_activation = torch.log
        def _cov_from_scale_rot(s, mod, R):
            L = build_scaling_rotation(mod * s, R)
            return strip_symmetric(L @ L.transpose(1, 2))
        self.covariance_activation      = _cov_from_scale_rot
        self.opacity_activation         = torch.sigmoid
        self._activation                = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.inverse_rho_activation     = inverse_sigmoid
        self.rotation_activation        = torch.nn.functional.normalize

        # build all tensors
        self.init_parameters(start_hair_style, num_gaussians)

    @torch.no_grad()
    def compute_root_normals(self, k=6):
        """
        Estimate a unit normal for each scalp root by fitting a plane to its
        k nearest-neighbour roots (PCA: smallest eigen-vector of the 3×3
        covariance).  Runs entirely on the GPU once neighbour indices exist.

        Parameters
        ----------
        k
            Number of neighbours to use.  Defaults to ``self.neighbor_k`` and
            is capped to the length of ``self.neighbor_idx`` that was built
            in :py:meth:`compute_root_adjacency`.

        Returns
        -------
        normals : torch.Tensor
            Shape ``(S, 3)``, one L2-normalised outward-pointing normal per
            root, **on the same device** as ``self.roots``.
        """
        # make sure neighbour lookup exists
        if not hasattr(self, "neighbor_idx"):
            self.compute_root_adjacency()

        # clip k to what we actually have
        k = min(k, self.neighbor_idx.shape[1])
        idx = self.neighbor_idx[:, :k]              # (S,k)  on device

        roots = self.scalp_roots                          # (S,3)
        nbrs  = roots[idx]                          # (S,k,3)

        # center neighbours about each root
        centerd = nbrs - roots.unsqueeze(1)         # (S,k,3)
        # 3×3 covariance of neighbourhood
        cov = torch.matmul(centerd.transpose(2,1), centerd) / float(k)   # (S,3,3)

        # eigen-decomposition (ascending eigenvalues)
        eigval, eigvec = torch.linalg.eigh(cov)      # eigvec[...,i] is i-th eigen-vector
        normals = eigvec[..., 0]                     # smallest λ → plane normal  (S,3)

        # orient outward: “outward” ≈ away from root-cloud centroid
        center = roots.mean(dim=0, keepdim=True)     # (1,3)
        to_center = roots - center                   # (S,3)
        flip = (normals * to_center).sum(dim=1) > 0  # True → pointing *inwards*
        normals[flip] = -normals[flip]

        return torch.nn.functional.normalize(normals, dim=1)   # (S,3)

    def init_parameters(
        self,
        start_hair_style,
        total_gaussians: int = 3000000,
        thick_perp: float = 5.645693247264717e-05,
        par_ratio: float = 10,
        dup_factor: int = 1
    ):
        """
        Initialise PERM latents and a fixed total of Gaussians,
        distributing them along strands relative to strand lengths.
        """
        device = self.roots.device
        S = self.num_strands

        # ── 0) PERM latent codes θ, β ────────────────────────────────────────
        if start_hair_style:
            self.theta = nn.Parameter(
                torch.as_tensor(start_hair_style["theta"], device=device)[None],
                requires_grad=True
            )
            self.beta = nn.Parameter(
                torch.as_tensor(start_hair_style["beta"], device=device)[None],
                requires_grad=True
            )
        else:
            self.theta = nn.Parameter(self.perm.theta_avg().to(device), requires_grad=True)
            self.beta  = nn.Parameter(self.perm.beta_avg().to(device),  requires_grad=True)
        self.theta_start = self.theta.detach().clone()
        self.beta_start  = self.beta.detach().clone()

        # ── 1) Decode reference strands & compute lengths ────────────────────
        with torch.no_grad():
            strands = self.perm(
                roots=self.get_roots_xyz[None],
                theta=self.theta, beta=self.beta
            )["strands"].position[0] / 100.0   # (S, V, 3)
            _, V, _ = strands.shape

            seg_len  = (strands[:,1:] - strands[:,:-1]).norm(dim=-1)  # (S, V-1)
            strand_L = seg_len.sum(dim=1)                             # (S,)
            self.register_buffer("strand_length", strand_L, persistent=False)

            # Distribute total_gaussians by length
            total_L    = strand_L.sum().item()
            raw_counts = strand_L / total_L * float(total_gaussians)
            K_list     = raw_counts.round().long().clamp(min=1).tolist()

            # Fix rounding drift
            drift = total_gaussians - sum(K_list)
            if drift > 0:
                idx_desc = torch.argsort(strand_L, descending=True)
                for i in range(drift):
                    K_list[idx_desc[i % S]] += 1
            elif drift < 0:
                idx_asc = torch.argsort(strand_L)
                for i in range(-drift):
                    sid = idx_asc[i % S]
                    if K_list[sid] > 1:
                        K_list[sid] -= 1

            # Build axial samples per strand
            s_list, sid_list = [], []
            for sid, K in enumerate(K_list):
                s_uniform = torch.linspace(0, 1, K, device=device)
                eps = 1e-6                       # stay away from 0/1 to avoid ±∞
                s_clamped = s_uniform.clamp(min=eps, max=1-eps)
                s_hat_init = torch.log(s_clamped) - torch.log1p(-s_clamped)
                s_list.append(s_hat_init)

                sid_list.append(torch.full((K,), sid, device=device, dtype=torch.long))

            offsets = torch.zeros(S+1, dtype=torch.long, device=device)
            offsets[1:] = torch.cumsum(torch.tensor(K_list, device=device), dim=0)
            M = int(offsets[-1])
            self.register_buffer("_strand_id",    torch.cat(sid_list), persistent=False)
            self.register_buffer("_gauss_offset", offsets,             persistent=False)
            self.num_gaussians = M

        # ── 2) Voronoi‐based per‐strand cage radius ─────────────────────────
        strand_R = torch.ones((self.num_strands,), device=device) * (self.global_strand_radii * self.global_scale)
        self.register_buffer("_strand_radius", strand_R, persistent=False)

        # ── 3) Gradient accumulators & 2D‐radii ────────────────────────────
        self.xyz_gradient_accum = torch.zeros((M,1), device=device)
        self.xyz_full_grad_accum = torch.zeros((M, 2), device=device) # Shape (M, 2) for xy grads
        self.denom              = torch.zeros_like(self.xyz_gradient_accum)
        self.register_buffer("max_radii2D",
                            torch.zeros((M,), device=device),
                            persistent=False)

        # ── 4) Initial centers & quaternions ───────────────────────────────
        centers, quats = [], []
        for sid in range(S):
            a, b = offsets[sid].item(), offsets[sid+1].item()
            s_uniform = torch.linspace(0,1,b-a,device=device)
            c, t = self._center_tangent_normal(strands[sid], s_uniform)
            centers.append(c)
            quats  .append(self._edge_dirs_to_quat(t))
        self._xyz      = torch.cat(centers,  0)  # (M,3)
        self._rotation = torch.cat(quats,    0)  # (M,4)

        # ── 5) Learnable per‐Gaussian tensors ───────────────────────────────
        self._s       = nn.Parameter(torch.cat(s_list), requires_grad=True)
        self._phi     = nn.Parameter(torch.empty((M,1), device=device).uniform_(0,2*math.pi),
                                    requires_grad=True)
        rho           = torch.rand((M,1), device=device)
        rho_hat       = self.inverse_rho_activation(rho)
        self._rho_hat = nn.Parameter(rho_hat, requires_grad=True)

        n_sh = (self.max_sh_degree+1)**2
        feats = torch.empty((M,1,n_sh), device=device).uniform_(-1,1).repeat(1,3,1)
        self._features_dc   = nn.Parameter(feats[:,:,:1].contiguous(), requires_grad=True)
        self._features_rest = nn.Parameter(feats[:,:,1:].contiguous(), requires_grad=True)
        self._opacity       = nn.Parameter(self.inverse_opacity_activation(
                                            torch.ones((M,1), device=device)),
                                        requires_grad=True)

        # ── 6) Default scale bases ──────────────────────────────────────────
        sigma_perp0 = thick_perp * self.global_scale
        sigma_par0  = sigma_perp0 * par_ratio * self.global_scale
        log0        = torch.tensor([sigma_perp0, sigma_par0, sigma_perp0],
                                device=device).log()
        self._scaling_base = nn.Parameter(log0.expand(M,3).clone(), requires_grad=True)
        self._scaling      = None

        # ── 7) ASG Initialization ─────────-────────────────────────────────
        self._features_asg = nn.Parameter(torch.zeros(M, self.max_asg_degree, device=device)).float()

        # ── 8) Uncertainty Initialization ─────────-────────────────────────────────
        p_target = 0.5
        logit_p  = math.log(p_target / (1.0 - p_target))   # ≈ 2.1972245
        gate_init = torch.zeros(M, self.max_asg_degree + 1, device=device)
        gate_init[:, 0].fill_(logit_p) 

        self._gate_logit = nn.Parameter(gate_init).float()

        # ── 9) Optional duplication ─────────────────────────────────────────
        if dup_factor > 1:
            orig_ids = self._strand_id
            self._strand_id = orig_ids.repeat(dup_factor)
            counts = torch.bincount(self._strand_id, minlength=S)
            new_off = torch.zeros(S+1, dtype=torch.long, device=device)
            new_off[1:] = counts.cumsum(0)
            self._gauss_offset = new_off

            def rep(x): return x.repeat(dup_factor, *[1]*(x.dim()-1))
            self._s             = nn.Parameter(rep(self._s),             True)
            self._phi           = nn.Parameter(rep(self._phi),           True)
            self._rho_hat       = nn.Parameter(rep(self._rho_hat),       True)
            self._features_dc   = nn.Parameter(rep(self._features_dc),   True)
            self._features_rest = nn.Parameter(rep(self._features_rest), True)
            self._features_asg  = nn.Parameter(rep(self._features_asg),  True)
            self._opacity       = nn.Parameter(rep(self._opacity),       True)
            self._scaling_base  = nn.Parameter(rep(self._scaling_base),  True)
            self._xyz           = rep(self._xyz)
            self._rotation      = rep(self._rotation)

            M = self._s.shape[0]
            self.num_gaussians = M
            self.xyz_gradient_accum = torch.zeros((M,1), device=device)
            self.xyz_full_grad_accum = torch.zeros((M,2), device=device)
            self.denom              = torch.zeros_like(self.xyz_gradient_accum)
            self.register_buffer("max_radii2D",
                                torch.zeros((M,), device=device),
                                persistent=False)

        # ── 10) One disk‑Gaussian per scalp root  --------------------------
        # We append S extra Gaussians (one per root) that represent
        # scalp‑attached “disks”.  They live in the same tensors as the
        # strand Gaussians; their indices are stored in self._scalp_id.
        N_scalp = S
        prev_M  = self.num_gaussians        # old total before adding disks
        new_M   = prev_M + N_scalp

        # a) centers & rotation (root position in *metres*, normal‑based quat)
        scalp_xyz  = self.scalp_roots / 100.0 + self.scalp_normals * self.scalp_offset
        scalp_quat = self._edge_dirs_to_quat(self.scalp_normals)      # (S,4)

        # NOTE: axial (s) and radial cage (rho/phi) are fixed and will be masked
        #       out of optimisation in training_setup().
        # c) axial / radial cage are fixed to 0
        scalp_s   = torch.full((N_scalp,), -10.0, device=device)          # sigmoid ≈ 0
        scalp_rho = torch.full((N_scalp,1), 0.0, device=device)
        scalp_phi = torch.full((N_scalp,1), 0.0, device=device)

        # d) color / opacity / ASG  — random like the strand init
        n_sh = (self.max_sh_degree + 1) ** 2
        f = torch.zeros((N_scalp,1,n_sh), device=device).uniform_(-1,1).repeat(1,3,1)
        scalp_dc   = f[:,:,:1].contiguous()
        scalp_rest = f[:,:,1:].contiguous()
        scalp_asg  = torch.zeros(N_scalp, self.max_asg_degree, device=device)
        scalp_opac = self.inverse_opacity_activation(1.0 * torch.ones((N_scalp,1), device=device))

        # e) scaling: use same default log σ, making them disk‑like later
        log0 = torch.tensor([self.global_strand_radii * self.global_scale, thick_perp * self.global_scale, self.global_strand_radii * self.global_scale],
                            device=device).log()
        scalp_scale = log0.expand(N_scalp,3).clone()
        self.scalp_radii = self.global_strand_radii * self.global_scale

        # f) gate init
        p_target = 0.5
        logit_p  = math.log(p_target / (1.0 - p_target))
        scalp_gate = torch.zeros(N_scalp, self.max_asg_degree + 1, device=device)
        scalp_gate[:,0].fill_(logit_p)

        # g) concatenate onto every tensor / parameter ----------------------
        self._xyz           = torch.cat((self._xyz,           scalp_xyz),  0)
        self._rotation      = torch.cat((self._rotation,      scalp_quat), 0)
        self._s             = nn.Parameter(torch.cat((self._s,             scalp_s),   0), True)
        self._rho_hat       = nn.Parameter(torch.cat((self._rho_hat,       scalp_rho), 0), True)
        self._phi           = nn.Parameter(torch.cat((self._phi,           scalp_phi), 0), True)
        self._features_dc   = nn.Parameter(torch.cat((self._features_dc,   scalp_dc),   0), True)
        self._features_rest = nn.Parameter(torch.cat((self._features_rest, scalp_rest), 0), True)
        self._features_asg  = nn.Parameter(torch.cat((self._features_asg,  scalp_asg), 0), True)
        self._opacity       = nn.Parameter(torch.cat((self._opacity,       scalp_opac), 0), True)
        self._scaling_base  = nn.Parameter(torch.cat((self._scaling_base,  scalp_scale),0),True)
        self._gate_logit.data = torch.cat((self._gate_logit.data, scalp_gate), 0)

        # h) expand gradient accumulators & radii buffers
        z = torch.zeros((N_scalp,1), device=device)
        z2 = torch.zeros((N_scalp,2), device=device)
        self.xyz_gradient_accum = torch.cat((self.xyz_gradient_accum, z), 0)
        self.xyz_full_grad_accum = torch.cat((self.xyz_full_grad_accum, z2), 0)
        self.denom              = torch.cat((self.denom,              z), 0)
        self.max_radii2D        = torch.cat((self.max_radii2D,
                                             torch.zeros(N_scalp, device=device)), 0)

        # update global count and record indices
        self.register_buffer(
            "_scalp_id",
            prev_M + torch.arange(S, device=device, dtype=torch.long),
            persistent=False
        )
        self.num_gaussians = new_M

        self.active_sh_degree = 0
        self.spatial_lr_scale = 1.0

        # diagnostics cached every frame for densifcation/pruning (all plain attributes, not nn.Parameter)
        self.strand_grad_kappa_accum = torch.zeros(M, 1, device=device)
        self.strand_grad_tau_accum   = torch.zeros(M, 1, device=device)
        self.strand_grad_denom       = torch.zeros(M, 1, device=device)
    
    def compute_root_adjacency(self):
        K_MAX = 32
        with torch.no_grad():
            # move to CPU for KD‐tree
            roots_cpu = self.roots.detach().cpu().numpy()           # (S,3)
            tree      = _sp.cKDTree(roots_cpu)
            # query K_MAX+1 because first neighbour is self
            _, idx_all = tree.query(roots_cpu, k=K_MAX + 1)         # (S, K_MAX+1)
            # drop the self‐match at index 0, keep up to K_MAX neighbours
            neighbor_idx = torch.from_numpy(idx_all[:, 1:]).long()  # (S, K_MAX)
            # register as a buffer so it’s on the right device & saved in state_dict
            self.register_buffer("neighbor_idx", neighbor_idx.to(self.roots.device),
                                 persistent=False)

    @torch.no_grad()
    def reset_gaussians_to_new_roots(
        self,
        new_roots: torch.Tensor,         # (S_new, 3) in metres
        new_radii: float = 1.0,
        total_gaussians: int = 2_000_000,
        thick_perp: float = 5.645693247264717e-05,
        par_ratio:   float = 20.0,
        k_neighbors: int   = 1,
        dup_factor:  int   = 1,
        uniform_strand_color: bool = False,    # ← NEW FLAG
        random_color: bool = False
    ):
        """
        Re-initialise the Gaussian scene on a new set of scalp roots.

        Parameters
        ----------
        uniform_strand_color
            If **True**, color SH coefficients, ASG features and opacity logits
            are stored *once per strand* (shape `(S_new, …)`); the usual attribute
            names (`_features_dc`, `_features_rest`, `_features_asg`, `_opacity`)
            are **re-used**.  The properties `get_features`, `get_asg_features`
            and `get_opacity` automatically broadcast them to `(M, …)` when
            accessed, so no other code must change.
        """
        device = self._xyz.device
        dtype  = self._xyz.dtype

        # Preemptively save relevant old scalp data
        old_scalp_xyz = self._xyz[self._scalp_id]
        old_scalp_rot = self._rotation[self._scalp_id]
        old_scalp_s = self._s[self._scalp_id]
        old_scalp_rho_hat = self._rho_hat[self._scalp_id]
        old_scalp_phi = self._phi[self._scalp_id]
        old_scalp_features_dc = self._features_dc[self._scalp_id]
        old_scalp_features_rest = self._features_rest[self._scalp_id]
        old_scalp_features_asg = self._features_asg[self._scalp_id]
        old_scalp_opacity = self._opacity[self._scalp_id]
        old_scalp_scaling_base = self._scaling_base[self._scalp_id]
        old_scalp_gate_logit = self._gate_logit[self._scalp_id]

        # Work *only* on Gaussians that belong to strands (ignore scalp disks)
        strand_idx    = self._strand_id                       # (M_strand,)
        M_old         = strand_idx.shape[0]                   # strands only

        dc_shape   = self._features_dc.shape[1:]              # (3, 1)
        rest_shape = self._features_rest.shape[1:]            # (3, n_rest)
        Ddc  = int(np.prod(dc_shape))
        Dr   = int(np.prod(rest_shape))
        Dasg = self._features_asg.shape[1]

        old_dc_flat   = self._features_dc  [strand_idx].reshape(M_old, Ddc)
        old_rest_flat = self._features_rest[strand_idx].reshape(M_old, Dr)
        old_asg_flat  = self._features_asg [strand_idx]                       # (M_old, Dasg)

        # ------------------------------------------------------------------ 1.  per-old-strand means
        S_old = self.num_strands
        strand_feats_color = torch.zeros((S_old, Ddc + Dr), device=device, dtype=dtype)
        strand_feats_asg    = torch.zeros((S_old, Dasg     ), device=device, dtype=dtype)
        counts              = torch.zeros((S_old,),        device=device, dtype=dtype)

        strand_feats_color.index_add_(0, strand_idx,torch.cat([old_dc_flat, old_rest_flat], dim=1))
        strand_feats_asg.index_add_(0, strand_idx, old_asg_flat)
        counts.index_add_(0, strand_idx, torch.ones_like(strand_idx, dtype=dtype))

        nz = counts > 0
        strand_feats_color[nz] /= counts[nz].unsqueeze(1)
        strand_feats_asg   [nz] /= counts[nz].unsqueeze(1)

        # ------------------------------------------------------------------ 2.  k-NN color transfer to new roots
        old_np = self.roots.cpu().numpy()
        new_np = new_roots.cpu().numpy()
        tree   = _sp.cKDTree(old_np)
        _, nearest_neigh = tree.query(new_np, k=k_neighbors)    # (S_new, k)

        S_new = new_roots.shape[0]
        new_color_per_strand = torch.zeros((S_new, Ddc + Dr), device=device, dtype=dtype)
        new_asg_per_strand    = torch.zeros((S_new, Dasg     ), device=device, dtype=dtype)

        for i in range(S_new):
            neigh = nearest_neigh[i]
            if neigh:
                new_color_per_strand[i] = strand_feats_color[neigh].mean(0)
                new_asg_per_strand   [i] = strand_feats_asg   [neigh].mean(0)
            else:                                  # degenerate – randomise
                new_color_per_strand[i].uniform_(-1, 1)
                new_asg_per_strand   [i].zero_()

        # ------------------------------------------------------------------ 3.  decode new strands & lengths
        out      = self.perm(roots=new_roots[None],
                            theta=self.theta, beta=self.beta)
        strands  = out["strands"].position[0] / 100.0      # (S_new, V, 3)
        seg_len  = (strands[:, 1:] - strands[:, :-1]).norm(dim=-1)
        strand_L = seg_len.sum(dim=1)                      # (S_new,)
        self.register_buffer("strand_length", strand_L, persistent=False)

        # ------------------------------------------------------------------ 4.  redistribute total_gaussians
        if total_gaussians is None:
            total_gaussians = M_old
        raw     = (strand_L / strand_L.sum()) * float(total_gaussians)
        K_list  = raw.round().long().clamp(min=1).tolist()
        drift   = total_gaussians - sum(K_list)
        if drift > 0:
            o = torch.argsort(strand_L, descending=True)
            for i in range(drift):
                K_list[o[i % S_new]] += 1
        elif drift < 0:
            o = torch.argsort(strand_L)
            for i in range(-drift):
                sid = o[i % S_new]
                if K_list[sid] > 1:
                    K_list[sid] -= 1

        # ------------------------------------------------------------------ 5.  build axial logits + strand-id lookup
        s_list, sid_list = [], []
        for sid, K in enumerate(K_list):
            s_uniform  = torch.linspace(0, 1, K, device=device, dtype=dtype)
            eps        = 1e-6
            s_clamped  = s_uniform.clamp(min=eps, max=1-eps)
            s_hat_init = torch.log(s_clamped) - torch.log1p(-s_clamped)
            s_list .append(s_hat_init)
            sid_list.append(torch.full((K,), sid, device=device, dtype=torch.long))

        ids     = torch.cat(sid_list)
        offsets = torch.zeros((S_new + 1,), dtype=torch.long, device=device)
        offsets[1:] = torch.cumsum(torch.tensor(K_list, device=device), 0)

        self._s           = nn.Parameter(torch.cat(s_list), requires_grad=True)
        self.register_buffer("_strand_id",    ids,      persistent=False)
        self.register_buffer("_gauss_offset", offsets,  persistent=False)
        self.num_gaussians = int(offsets[-1])

        # ------------------------------------------------------------------ 6.  per-strand cage radius
        strand_R = torch.ones((S_new,), device=device, dtype=dtype) * new_radii * self.global_scale
        self.register_buffer("_strand_radius", strand_R, persistent=False)

        # ------------------------------------------------------------------ 7.  centers & quats
        centers, quats = [], []
        for sid in range(S_new):
            a, b = offsets[sid].item(), offsets[sid+1].item()
            if b > a:
                ss = torch.linspace(0, 1, b - a, device=device, dtype=dtype)
                c, t = self._center_tangent_normal(strands[sid], ss)
                centers.append(c);  quats.append(self._edge_dirs_to_quat(t))
        self._xyz      = torch.cat(centers, 0)
        self._rotation = torch.cat(quats,   0)

        # ------------------------------------------------------------------ 8.  initialise color / opacity / ASG
        if uniform_strand_color:
            # -------- one vector per strand (shape (S_new, …)) ---------------
            self._features_dc   = nn.Parameter(
                new_color_per_strand[:, :Ddc].view(S_new, 3, 1),
                requires_grad=True
            )
            self._features_rest = nn.Parameter(
                new_color_per_strand[:, Ddc:].view(S_new, 3, Dr // 3).contiguous(),
                requires_grad=True
            )
            self._features_asg  = nn.Parameter(
                new_asg_per_strand,                       # (S_new, Dasg)
                requires_grad=True
            )
            self._opacity       = nn.Parameter(
                self.inverse_opacity_activation(
                    torch.ones((S_new, 1), device=device, dtype=dtype)
                ),
                requires_grad=True
            )
            self.uniform_strand_color = True
        else:
            # ---------- one vector per Gaussian (legacy behaviour) -----------
            color = new_color_per_strand[self._strand_id]   # (M_new, Ddc+Dr)
            asg    = new_asg_per_strand   [self._strand_id]   # (M_new, Dasg)

            if random_color:
                color = torch.empty((len(color), 1, 1 + (Dr // 3)), device=device).uniform_(-1,1).repeat(1,3,1)

                dc_flat = color[:, :, 0].reshape(self.num_gaussians, 3)
                rs_flat = color[:, :, 1:].reshape(self.num_gaussians, Dr)
            else:
                dc_flat = color[:, :Ddc]
                rs_flat = color[:, Ddc:]

            self._features_dc   = nn.Parameter(
                dc_flat.view(self.num_gaussians, 3, 1),
                requires_grad=True
            )
            self._features_rest = nn.Parameter(
                rs_flat.view(self.num_gaussians, 3, Dr // 3).contiguous(),
                requires_grad=True
            )
            self._features_asg  = nn.Parameter(asg,  requires_grad=True)
            self._opacity       = nn.Parameter(
                self.inverse_opacity_activation(
                    torch.ones((self.num_gaussians, 1), device=device, dtype=dtype)
                ),
                requires_grad=True
            )
            self.uniform_strand_color = False

        # ------------------------------------------------------------------ 9.  other learnables
        # initialise gate *logits* (column 0) so that sigmoid(logit) ≈ N(0.9, 0.35²) in [0,1]
        p_target = 0.5
        logit_p  = math.log(p_target / (1.0 - p_target))   # ≈ 2.1972245
        gate_init = torch.zeros(self.num_gaussians, self.max_asg_degree + 1, device=device)
        gate_init[:, 0].fill_(logit_p) 

        self._gate_logit = nn.Parameter(gate_init).float()


        self._phi      = nn.Parameter(
            torch.empty((self.num_gaussians, 1), device=device, dtype=dtype)
                .uniform_(0, 2*math.pi),
            requires_grad=True
        )
        rho            = torch.rand((self.num_gaussians, 1), device=device, dtype=dtype)
        self._rho_hat  = nn.Parameter(self.inverse_rho_activation(rho), True)

        sigma_p = thick_perp * self.global_scale
        sigma_l = sigma_p * par_ratio
        log0    = torch.tensor([sigma_p, sigma_l, sigma_p],
                            device=device, dtype=dtype).log()
        self._scaling_base = nn.Parameter(
            log0.expand(self.num_gaussians, 3).clone(), True
        )
        self._scaling = None

        # ------------------------------------------------------------------ 10.  accumulators
        self.xyz_gradient_accum = torch.zeros((self.num_gaussians, 1), device=device, dtype=dtype)
        self.xyz_full_grad_accum = torch.zeros((self.num_gaussians, 2), device=device, dtype=dtype)
        self.denom              = torch.zeros_like(self.xyz_gradient_accum)
        self.register_buffer("max_radii2D",
                            torch.zeros((self.num_gaussians,), device=device, dtype=dtype),
                            persistent=False)

        # ------------------------------------------------------------------ 11.  optional duplication block (unchanged)
        if dup_factor > 1:
            orig_ids = self._strand_id
            self._strand_id = orig_ids.repeat(dup_factor)
            counts = torch.bincount(self._strand_id, minlength=S_new)
            new_off = torch.zeros(S_new + 1, dtype=torch.long, device=device)
            new_off[1:] = counts.cumsum(0)
            self._gauss_offset = new_off

            def rep(x): return x.repeat(dup_factor, *[1]*(x.dim()-1))
            self._s            = nn.Parameter(rep(self._s),            True)
            self._phi          = nn.Parameter(rep(self._phi),          True)
            self._rho_hat      = nn.Parameter(rep(self._rho_hat),      True)
            self._features_dc  = nn.Parameter(rep(self._features_dc),  True)
            self._features_rest= nn.Parameter(rep(self._features_rest),True)
            self._features_asg = nn.Parameter(rep(self._features_asg), True)
            self._opacity      = nn.Parameter(rep(self._opacity),      True)
            self._scaling_base = nn.Parameter(rep(self._scaling_base), True)
            self._xyz          = rep(self._xyz)
            self._rotation     = rep(self._rotation)

            self.num_gaussians = self._s.shape[0]
            self.xyz_gradient_accum = torch.zeros((self.num_gaussians, 1), device=device, dtype=dtype)
            self.xyz_full_grad_accum = torch.zeros((self.num_gaussians, 2), device=device, dtype=dtype)
            self.denom              = torch.zeros_like(self.xyz_gradient_accum)
            self.register_buffer("max_radii2D",
                                torch.zeros((self.num_gaussians,), device=device, dtype=dtype),
                                persistent=False)
        
        del self.neighbor_idx
        self.roots       = new_roots.to(device, dtype=dtype)
        self.compute_root_adjacency()

        N_scalp = len(old_scalp_xyz)
        prev_M  = self.num_gaussians
        new_M   = prev_M + N_scalp

        self.num_gaussians = new_M
        self.num_strands = S_new

        if uniform_strand_color:
            old_scalp_opacity = torch.ones_like(old_scalp_opacity).to(device)

        # g) concatenate onto every tensor / parameter ----------------------
        self._xyz           = torch.cat((self._xyz,           old_scalp_xyz),  0)
        self._rotation      = torch.cat((self._rotation,      old_scalp_rot), 0)
        self._s             = nn.Parameter(torch.cat((self._s,             old_scalp_s),   0), True)
        self._rho_hat       = nn.Parameter(torch.cat((self._rho_hat,       old_scalp_rho_hat), 0), True)
        self._phi           = nn.Parameter(torch.cat((self._phi,           old_scalp_phi), 0), True)
        self._features_dc   = nn.Parameter(torch.cat((self._features_dc,   old_scalp_features_dc),   0), True)
        self._features_rest = nn.Parameter(torch.cat((self._features_rest, old_scalp_features_rest), 0), True)
        self._features_asg  = nn.Parameter(torch.cat((self._features_asg,  old_scalp_features_asg), 0), True)
        self._opacity       = nn.Parameter(torch.cat((self._opacity,       old_scalp_opacity), 0), True)
        self._scaling_base  = nn.Parameter(torch.cat((self._scaling_base,  old_scalp_scaling_base),0),True)
        self._gate_logit.data = torch.cat((self._gate_logit.data, old_scalp_gate_logit), 0)

        # h) expand gradient accumulators & radii buffers
        z = torch.zeros((N_scalp,1), device=device)
        z2 = torch.zeros((N_scalp,2), device=device)
        self.xyz_gradient_accum = torch.cat((self.xyz_gradient_accum, z), 0)
        self.xyz_full_grad_accum = torch.cat((self.xyz_full_grad_accum, z2), 0)
        self.denom              = torch.cat((self.denom,              z), 0)
        self.max_radii2D        = torch.cat((self.max_radii2D,
                                             torch.zeros(N_scalp, device=device)), 0)

        self.strand_grad_kappa_accum = torch.zeros(self.num_gaussians - N_scalp, 1).to(device)
        self.strand_grad_tau_accum = torch.zeros(self.num_gaussians - N_scalp, 1).to(device)
        self.strand_grad_denom = torch.zeros(self.num_gaussians - N_scalp, 1).to(device)

        self.active_sh_degree = 0
        self.spatial_lr_scale = 1.0
        self.register_buffer(
            "_scalp_id",
            prev_M + torch.arange(N_scalp, device=device, dtype=torch.long),
            persistent=False
        )

    @torch.no_grad()
    def reassign_gaussians_to_new_roots(
        self,
        new_roots: torch.Tensor,       # (S_new, 3) the newly densified root positions (in metres)
        new_radii: float = 1.0,        # optional, if you want to reset strand radii
        k_neighbors: int = 6,          # we want exactly the 6 nearest new roots for each old root
        chunk_size: int = 5000
    ):
        """
        Re‐assign each of the M Gaussians to the closest of its old‐root’s k_neighbors
        among the new_roots.  We do this in chunks to avoid O(M·k·V) memory blowup.

        Steps:
         1. Keep track of old roots (self.roots) and count S_old = self.num_strands.
         2. Build a small KD‐tree on new_roots (CPU) and query each old root for its
            k_neighbors nearest‐new‐roots → a (S_old, k_neighbors) array of indices.
         3. For each Gaussian (there are M of them), look up which old root it “came from”
            (orig_id = self._strand_id), then gather that old root’s k_neighbors
            candidate new‐root indices.
         4. Decode all new strands once (with new_roots, same θ/β), then—for every Gaussian—
            find which of its k_neighbors (and which vertex along that neighbor‐strand)
            is closest in Euclidean space.  Assign accordingly.
         5. Rebuild self._strand_id (length M) by taking each Gaussian’s best new‐root index,
            rebuild prefix sums, recompute cage radii, strand_length, etc.
        """
        device = self._xyz.device

        # 1) Keep references to “old” root set:
        old_roots = self.roots.clone()    # (S_old, 3)
        S_old = old_roots.shape[0]
        M = self._s.shape[0]              # total number of Gaussians

        # 2) Build KD‐tree on new_roots (move to CPU numpy for cKDTree)
        new_roots_cpu = new_roots.detach().cpu().numpy()  # (S_new, 3)
        old_roots_cpu = old_roots.detach().cpu().numpy()  # (S_old, 3)

        #    Query each old root for its k_neighbors nearest among new_roots
        tree = _sp.cKDTree(new_roots_cpu)
        #    distances not used here, we only need the indices
        _, nn_idx = tree.query(old_roots_cpu, k=k_neighbors)  # (S_old, k_neighbors)

        #    Now nn_idx[i] is an array of k_neighbors indices into new_roots_cpu
        #    For each old‐root i, the new‐roots at those indices form its “candidates.”
        #    We turn this into a torch.LongTensor for indexing on GPU:
        cand_per_old = torch.from_numpy(nn_idx.astype(np.int64)).to(device)  # (S_old, k_neighbors)

        # 3) For each Gaussian (index in [0..M-1]), look up which old root it belonged to:
        orig_id = self._strand_id       # shape (M,), each entry ∈ [0..S_old-1]

        #    Now “cand_ids_for_each_gaussian” will be shape (M, k_neighbors):
        #    cand_ids_for_each_gaussian[g] = cand_per_old[orig_id[g]]
        cand_ids = cand_per_old[orig_id]  # (M, k_neighbors)

        # 4) Decode all new strands once under (new_roots, same θ, β):
        #    We assume self.perm(...) returns a dict with ["strands"].position = (1, S_new, V, 3).
        strands_dict = self.perm(
            roots=new_roots[None],   # shape (1, S_new, 3)
            theta=self.theta,
            beta=self.beta
        )
        new_strands = strands_dict["strands"].position[0] / 100.0  # (S_new, V, 3)
        V = new_strands.shape[1]
        Kp1 = k_neighbors

        # Prepare output buffers:
        #   - best_root_idx_for_gaussian: which of the k_neighbors is chosen
        #   - best_vert_idx_for_gaussian: which vertex along that chosen neighbor‐strand
        best_root_idx = torch.empty((M,), dtype=torch.long, device=device)
        best_vert_idx = torch.empty((M,), dtype=torch.long, device=device)

        # We process Gaussians in chunks of size chunk_size to avoid gigantic temporary tensors:
        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            idx_slice = slice(start, end)        # indices [start..end-1]
            C = end - start                      # chunk size

            #   a) For this chunk, gather the candidate‐strand indices:
            #      cand_ids_chunk: shape (C, k_neighbors)
            cand_ids_chunk = cand_ids[idx_slice]

            #   b) Gather the new_strand positions for all these candidates:
            #      First flatten out, then reshape.  In detail:
            #       - cand_ids_chunk.view(-1) is size (C*k_neighbors,)
            #       - new_strands[cand_ids_chunk.view(-1)] is (C*k_neighbors, V, 3)
            flat = new_strands[cand_ids_chunk.view(-1)]          # (C*k_neighbors, V, 3)
            #       - reshape into (C, k_neighbors, V, 3)
            cand_str = flat.view(C, Kp1, V, 3)                   # (C, k_neighbors, V, 3)

            #   c) Extract each Gaussian’s current center (M×3).  We already have self._xyz (M,3):
            xyz_chunk = self._xyz[idx_slice].view(C, 1, 1, 3)     # (C, 1, 1, 3)

            #   d) Compute squared distances from each Gaussian center to every point on
            #      each candidate strand:
            #       → (cand_str - xyz_chunk)² summed over last dim → (C, k_neighbors, V)
            d2 = ((cand_str - xyz_chunk) ** 2).sum(dim=-1)         # (C, k_neighbors, V)

            #   e) Flatten the last two dims → (C, k_neighbors * V), do argmin along that axis:
            d2_flat = d2.view(C, Kp1 * V)                         # (C, k_neighbors*V)
            arg_flat = d2_flat.argmin(dim=1)                      # (C,)

            #   f) Decode which neighbor‐strand (0..k_neighbors-1) and which vertex (0..V-1):
            best_root_idx[idx_slice] = arg_flat // V               # which of the k_neighbors
            best_vert_idx[idx_slice] = arg_flat % V                # which vertex along that strand

        # 5) Now build the new strand assignment:
        #    Each Gaussian’s new strand ID is:
        #      new_strand = cand_ids[g, best_root_idx[g]]
        #    And its new axial s ∈ [0,1] is best_vert_idx/ (V-1).
        new_strand_id = cand_ids[torch.arange(M, device=device), best_root_idx]  # (M,)
        new_s         = best_vert_idx.to(torch.float32).div(V - 1)                # (M,)

        # 6) Overwrite self._strand_id and rebuild prefix‐sum offsets:
        self._strand_id = new_strand_id
        counts = torch.bincount(new_strand_id, minlength=new_roots.shape[0])     # (S_new,)
        offsets = torch.zeros((new_roots.shape[0] + 1,), dtype=torch.long, device=device)
        offsets[1:] = counts.cumsum(dim=0)
        self._gauss_offset = offsets  # (S_new+1,)

        # 7) Update self.roots + self.num_strands
        self.roots = new_roots.to(device)
        self.num_strands = new_roots.shape[0]

        # 8) Recompute per‐strand radii if desired:
        self.global_strand_radii = new_radii * self.global_scale
        strand_R = torch.ones((self.num_strands,), device=device) * (new_radii)
        self.register_buffer("_strand_radius", strand_R, persistent=False)

        # 9) Recompute each strand’s total length:
        seg_len   = (new_strands[:, 1:] - new_strands[:, :-1]).norm(dim=-1)  # (S_new, V-1)
        strand_L  = seg_len.sum(dim=1)                                        # (S_new,)
        self.register_buffer("strand_length", strand_L, persistent=False)

        # 10) Jitter colors if needed (same as before):
        # n_sh = (self.max_sh_degree + 1) ** 2
        # feats = torch.empty((M, 1, n_sh), device=device).uniform_(-1, 1).repeat(1, 3, 1)
        # self._features_dc.data   = feats[:, :, :1].transpose(1, 2).contiguous()
        # self._features_rest.data = feats[:, :, 1:].transpose(1, 2).contiguous()
        self._opacity.data = self.inverse_opacity_activation(torch.ones((M,1), device=device))

        # 11) Overwrite axial weights & call update_xyz_rot_scale:
        self._s = nn.Parameter(new_s, requires_grad=True)
        self.update_xyz_rot_scale(new_strands)
        self.compute_root_adjacency()

    @staticmethod
    def _project_onto_polyline(pts: torch.Tensor,
                               pole: torch.Tensor,
                               s_param: torch.Tensor,
                               eps: float = 1e-8):
        """
        Differentiable projection of N points onto a V-vertex polyline, 
        using only each point’s own segment ±1 neighbors.

        Args:
            pts      : (N,3) query points
            pole     : (V,3) polyline vertices
            s_param  : (N,) fractional position along [0,1] of each query
        Returns:
            s_best   : (N,) scalar in [0,1]
            P_best   : (N,3) closest points on the polyline
        """
        N = pts.shape[0]
        V = pole.shape[0]
        # 1) Compute each Gaussian’s “home” segment index
        seg_f = (s_param * (V-1)).floor().long().clamp(0, V-2)  # (N,)
        # neighbors: seg0, seg-1, seg+1
        segs = torch.stack([
            seg_f.clamp(0, V-2),
            (seg_f - 1).clamp(0, V-2),
            (seg_f + 1).clamp(0, V-2),
        ], dim=1)  # (N,3)

        # 2) Gather endpoints A and B for each of the 3 candidate segments
        A = pole[segs]          # (N,3,3): 3 segments per point
        B = pole[segs+1]        # (N,3,3)

        # 3) Vector from A→B and A→P
        AB    = B - A                              # (N,3,3)
        AP    = pts.unsqueeze(1) - A               # (N,3,3)
        AB2   = (AB*AB).sum(dim=2).clamp(min=eps)   # (N,3)

        # 4) Project AP onto AB: t = (AP⋅AB)/(AB⋅AB), clamp to [0,1]
        t     = (AP*AB).sum(dim=2) / AB2           # (N,3)
        t_cl  = t.clamp(0.0, 1.0).unsqueeze(2)     # (N,3,1)

        # 5) Compute the 3 candidate projections
        P_proj = A + AB * t_cl                     # (N,3,3)

        # 6) Pick the closest one
        d2    = ((pts.unsqueeze(1) - P_proj)**2).sum(dim=2)  # (N,3)
        idx   = d2.argmin(dim=1)                             # (N,)

        # 7) Gather best t and segment index
        t_best   = t.gather(1, idx.unsqueeze(1)).squeeze(1)  # (N,)
        seg_best = segs.gather(1, idx.unsqueeze(1)).squeeze(1)  # (N,)

        # 8) Convert to global s ∈ [0,1]
        s_best = (seg_best.float() + t_best) / (V-1)

        # 9) Gather the best projected point
        P_best = P_proj.gather(1,
                   idx.view(-1,1,1).expand(-1,1,3)
                 ).squeeze(1)                                 # (N,3)

        return s_best, P_best

    def _center_tangent_normal(self, pts: torch.Tensor, s: torch.Tensor):
        """
        Given one strand’s vertices `pts` (V, 3) and axial weights `s` (K,),
        return the interpolated centers and *unit* tangents at those points.
        """
        V   = pts.shape[0]
        t   = s * (V - 1)
        idx0 = torch.clamp(t.floor().long(), 0, V - 2)
        idx1 = idx0 + 1

        w    = (t - idx0.float()).unsqueeze(1)            # (K, 1)
        P0, P1 = pts[idx0], pts[idx1]

        center  = (1 - w) * P0 + w * P1                   # linear interp.
        tangent = torch.nn.functional.normalize(P1 - P0,  # **was .normalize()**
                                                dim=1,
                                                eps=1e-9)
        return center, tangent
    
    def gidx_of_strand(self, k: int):
        """Return tensor of Gaussian indices belonging to strand k"""
        a, b = self._gauss_offset[k:k+2].tolist()
        return torch.arange(a, b, device=self._s.device)

    def slice_of_strand(self, k: int):
        """Return slice(start,end) so big tensors can be indexed directly"""
        a, b = self._gauss_offset[k:k+2].tolist()
        return slice(a, b)

    def capture(self):
        """
        Packs all essential model state into a dictionary for checkpointing.
        """
        # Consolidate all per-Gaussian nn.Parameters
        per_gauss_params = {
            "s": self._s,
            "rho_hat": self._rho_hat,
            "phi": self._phi,
            "scaling_base": self._scaling_base,
            "gate_logit": self._gate_logit,
        }
        # Add appearance parameters based on the current mode
        if self.uniform_strand_color:
             per_strand_params = {
                "features_dc": self._features_dc,
                "features_rest": self._features_rest,
                "features_asg": self._features_asg,
                "opacity": self._opacity,
             }
             per_gauss_params.update(per_strand_params)
        else:
            per_gauss_params.update({
                "features_dc": self._features_dc,
                "features_rest": self._features_rest,
                "features_asg": self._features_asg,
                "opacity": self._opacity,
            })

        return {
            # Core learnable parameters (PERM latents)
            "theta": self.theta,
            "beta": self.beta,
            
            # All per-Gaussian or per-strand parameters
            "gauss_params": per_gauss_params,
            
            # Optimizer state
            "optimizer": self.optimizer.state_dict(),
            
            # Non-learnable state and buffers
            "active_sh_degree": self.active_sh_degree,
            "uniform_strand_color": self.uniform_strand_color,
            "roots": self.roots,
            "_strand_id": self._strand_id,
            "_scalp_id": self._scalp_id,
            "_gauss_offset": self._gauss_offset,
            "_strand_radius": self._strand_radius,
            "strand_length": self.strand_length,
            
            # Densification accumulators
            "xyz_gradient_accum": self.xyz_gradient_accum,
            "xyz_full_grad_accum": self.xyz_full_grad_accum,
            "denom": self.denom,
            "max_radii2D": self.max_radii2D,
            "strand_grad_kappa_accum": self.strand_grad_kappa_accum,
            "strand_grad_tau_accum": self.strand_grad_tau_accum,
            "strand_grad_denom": self.strand_grad_denom,
        }

    def restore(self, model_args, training_args, extra_parameters=None):
        """
        Unpacks a state dictionary to restore the model.
        """
        device = self._xyz.device

        num_gauss_to_restore = model_args["gauss_params"]["s"].shape[0]

        # Restore simple attributes and training state
        self.active_sh_degree = model_args.get("active_sh_degree", 0)
        self.uniform_strand_color = model_args.get("uniform_strand_color", False)

        # Restore PERM latents
        self.theta.data = model_args["theta"].data
        self.beta.data = model_args["beta"].data

        # Restore all per-Gaussian/per-strand parameters
        gauss_params = model_args["gauss_params"]
        for name, param in gauss_params.items():
            getattr(self, f"_{name}").data = param.data

        # Restore non-learnable buffers
        self.roots = model_args["roots"]
        self.register_buffer("_strand_id", model_args["_strand_id"])
        self.register_buffer("_scalp_id", model_args["_scalp_id"])
        self.register_buffer("_gauss_offset", model_args["_gauss_offset"])
        self.register_buffer("_strand_radius", model_args["_strand_radius"])
        self.register_buffer("strand_length", model_args["strand_length"])
        
        # Restore densification accumulators safely
        self.xyz_gradient_accum = model_args.get("xyz_gradient_accum", torch.zeros_like(self._xyz.data[:,:1]))
        self.xyz_full_grad_accum = model_args.get("xyz_full_grad_accum", torch.zeros((num_gauss_to_restore, 2), device=device))
        self.denom = model_args.get("denom", torch.zeros_like(self._xyz.data[:,:1]))
        self.max_radii2D = model_args.get("max_radii2D", torch.zeros(self._xyz.shape[0], device=device))
        
        num_gauss = self._s.shape[0]
        self.strand_grad_kappa_accum = model_args.get("strand_grad_kappa_accum", torch.zeros((num_gauss, 1), device=device))
        self.strand_grad_tau_accum = model_args.get("strand_grad_tau_accum", torch.zeros((num_gauss, 1), device=device))
        self.strand_grad_denom = model_args.get("strand_grad_denom", torch.zeros((num_gauss, 1), device=device))
        
        # Rebuild optimizer and load its state
        self.training_setup(training_args, extra_parameters)
        # self.optimizer.load_state_dict(model_args["optimizer"])

        # Recompute derived state
        self.num_gaussians = self._s.shape[0]
        self.num_strands = self.roots.shape[0]
        self.compute_root_adjacency()

    @property
    def get_roots_xyz(self):
        """Scalp-root locations (S,3) in metres."""
        return self.roots

    @property
    def get_xyz(self):
        """Current centers (M,3) – recomputed every frame."""
        return self._xyz

    @property
    def get_rotation(self):
        """Unit quaternions (M,4) that align +Y with the strand tangent."""
        return self.rotation_activation(self._rotation)

    @property
    def get_scaling(self):
        """
        Positive sigma values (M,3) in world units,
        with the longitudinal sigma‖ clamped to half the strand’s length.
        """
        raw = self.scaling_activation(self._scaling_base)  # (M,3)

        raw_strand = raw[:len(self._strand_id)]
        raw_scalp = raw[self._scalp_id]

        perp_strand = raw_strand[:, 0] # torch.minimum(raw[:, 0], self._strand_radius.mean()) # FIX ME
        paral_strand = torch.maximum(raw_strand[:, 1], perp_strand * 1.5)  # clamp sigma‖
        clamped_strand = torch.stack((perp_strand, paral_strand, perp_strand), dim=1)

        radii = torch.tensor(self.scalp_radii).to(raw.device)

        perp_scalp = torch.minimum(raw_scalp[:, 1], torch.tensor(5.645693247264717e-05 * 1.25).to(raw_scalp.device))
        paral_scalp_x = torch.minimum(raw_scalp[:, 0], radii * 1.25)
        paral_scalp_z = torch.minimum(raw_scalp[:, 2], radii * 1.25)
        clamped_scalp = torch.stack((paral_scalp_x, perp_scalp, paral_scalp_z), dim=1)

        return torch.cat([clamped_strand, clamped_scalp])

    @property
    def get_features(self):
        """
        Returns per-Gaussian features, with optional strand-level uniformity.
        """
        if getattr(self, "uniform_strand_color", False):
            # Mapping logic:
            #   • strand‑attached Gaussians → broadcast per‑strand vectors
            #   • scalp Gaussians          → use their own dedicated rows
            device = self._strand_id.device
            M      = self.num_gaussians
            S      = self.num_strands
            N_scalp = self._scalp_id.shape[0]

            mapping = torch.empty((M,), dtype=torch.long, device=device)

            # 1) strands: map to their strand idx (0‥S‑1)
            mapping[:len(self._strand_id)] = self._strand_id

            # 2) scalp: their SH rows were appended after the S strand rows
            scalp_feat_rows = S + torch.arange(N_scalp, device=device)   # (N_scalp,)
            mapping[len(self._strand_id):] = scalp_feat_rows

            dc   = self._features_dc  [mapping]         # (M,1,n_dc)
            rest = self._features_rest[mapping]
            return torch.cat((dc, rest), dim=-1)
        else:
            # tensors already per‑Gaussian
            return torch.cat((self._features_dc, self._features_rest), dim=-1)

    @property
    def get_asg_features(self):
        if getattr(self, "uniform_strand_color", False):
            device = self._strand_id.device
            M      = self.num_gaussians
            S      = self.num_strands
            N_scalp = self._scalp_id.shape[0]

            mapping = torch.empty((M,), dtype=torch.long, device=device)
            mapping[:len(self._strand_id)] = self._strand_id
            scalp_feat_rows = S + torch.arange(N_scalp, device=device)
            mapping[len(self._strand_id):]  = scalp_feat_rows

            out = self._features_asg[mapping].clone()
            return out
        else:
            out = self._features_asg.clone()
            return out

    @property
    def get_opacity(self):
        if getattr(self, "uniform_strand_color", False):
            device = self._strand_id.device
            M      = self.num_gaussians
            S      = self.num_strands
            N_scalp = self._scalp_id.shape[0]

            mapping = torch.empty((M,), dtype=torch.long, device=device)
            mapping[:len(self._strand_id)] = self._strand_id
            scalp_feat_rows = S + torch.arange(N_scalp, device=device)
            mapping[len(self._strand_id):]  = scalp_feat_rows
            return torch.sigmoid(self._opacity[mapping])
        else:
            return torch.sigmoid(self._opacity)

    @property
    def get_axial_weight(self):           # s in [0,1]
        return torch.sigmoid(self._s)
    
    @property
    def get_gate_per_gaussian(self):           
        return self._gate_logit

    @property
    def get_scaling_with_3D_filter(self):
        scales = self.get_scaling
        
        scales = torch.square(scales) + torch.square(self.filter_3D)
        scales = torch.sqrt(scales)
        return scales
    
    @property
    def get_opacity_with_3D_filter(self):
        opacity = self.get_opacity
        # apply 3D filter
        scales = self.get_scaling
        
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)
        
        scales_after_square = scales_square + torch.square(self.filter_3D) 
        det2 = scales_after_square.prod(dim=1) 
        coef = torch.sqrt(det1 / det2)

        out = opacity * coef[..., None]
        out[self._scalp_id] = opacity[self._scalp_id]
        return out

    @property
    def get_radial_uv(self):
        """
        Return (u,v) offsets in metres – already clamped to max_radius (= m/2).
        """
        rho = torch.sigmoid(self._rho_hat)
        return self._max_radius * torch.cat([rho * torch.cos(self._phi),
                                            rho * torch.sin(self._phi)], dim=1)

    @torch.no_grad()
    def set_scalp_opacity(
            self,
            scalp_can,
            solid_frac:  float = 0.35,   # inner “solid” core  (0‥1)
            falloff_frac: float = 0.30,  # width of fading rim (0‥1)
            min_alpha:    float = 0.0,   # opacity at/after rim’s outer edge
            clamp_center: float = 1.0):  # opacity inside the solid core
        """
        Fade scalp–disk Gaussians based on distance *d* from the highest-Y
        root (the “crown”):

                        α(d) = clamp_center          for d <= R_solid
                                lerp(clamp_center→min_alpha)
                                                    for R_solid < d < R_fade
                                min_alpha            for d >= R_fade

        with
            R_max   = max_i‖x_i − x_crown‖
            R_solid = solid_frac   · R_max
            R_fade  = (1-falloff_frac) · R_max  (fade band starts here)

        Requirements: `solid_frac + falloff_frac ≤ 1.0`.  If they overlap,
        the function shrinks the fade band so it still fits.

        The logits for scalp rows in `self._opacity` are overwritten so the
        change is immediate but remains differentiable.
        """
        if not hasattr(self, "_scalp_id") or self._scalp_id.numel() == 0:
            return

        assert 0.0 <= solid_frac <= 1.0
        assert 0.0 <= falloff_frac <= 1.0

        device      = self._opacity.device
        scalp_idx   = self._scalp_id                       # (N_sc,)
        scalp_xyz   = scalp_can                 # (N_sc,3)

        # — 1. find crown (highest-Y) —
        crown_id    = torch.argmax(scalp_xyz[:, 1])
        crown_pos   = scalp_xyz[crown_id]                  # (3,)

        # — 2. radii —
        dists   = torch.norm(scalp_xyz - crown_pos, dim=1) # (N_sc,)
        R_max   = dists.max()
        if R_max <= 1e-6:
            return                                         # degenerate

        R_solid = solid_frac * R_max
        R_fade  = max(R_solid, R_max * (1.0 - falloff_frac))

        # — 3. opacity profile —
        alpha = torch.full_like(dists, min_alpha)          # start at min
        solid_mask = dists <= R_solid
        fade_mask  = (dists > R_solid) & (dists < R_fade)

        alpha[solid_mask] = clamp_center                  # full inside core
        if fade_mask.any():
            t = (dists[fade_mask] - R_solid) / (R_fade - R_solid)
            alpha[fade_mask] = clamp_center * (1.0 - t) + min_alpha * t

        # — 4. write back as logits —
        logits_new = self.inverse_opacity_activation(alpha.unsqueeze(1))
        self._opacity.data[-len(scalp_idx):] = logits_new

    # ------------------ full 3×3 covariance -------------------------
    def get_covariance(self, scaling_modifier=1.0):
        """
        Return the symmetric 3×3 covariance matrix for each Gaussian,
        ready for the renderer.  The optional `scaling_modifier`
        is used by the mip-map style coarse-to-fine schedule.
        """
        return self.covariance_activation(self.get_scaling,
                                        scaling_modifier,
                                        self._rotation)

    def oneupSHdegree(self):
        """Unlock the next SH band (same rule as 3D-GS)."""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1


    def training_setup(self, training_args, extra_parameters=None):
        """
        Build the Adam optimiser with separate groups for every learnable
        tensor, assign sensible learning-rates, and create per-latent LR
        schedules for θ and β.
        """
        self.spatial_lr_scale = 5.0            # keep same scaling factor
        self.percent_dense    = training_args.percent_dense

        # ------------------------------------------------------------
        # param-groups with individual learning rates
        l = [
            # PERM latents
            {"params": [self.theta], "lr": training_args.theta_lr_init, "name": "theta"},
            {"params": [self.beta],  "lr": training_args.beta_lr_init,  "name": "beta" },

            # Gaussian appearance
            {"params": [self._features_dc],   "lr": training_args.feature_lr,           "name": "f_dc"},
            {"params": [self._features_rest], "lr": training_args.feature_lr / 20.0,    "name": "f_rest"},
            {'params': [self._features_asg],  'lr': training_args.feature_lr,           "name": "f_asg"},
            {"params": [self._opacity],       "lr": training_args.opacity_lr,           "name": "opacity"},

            # Gaussian shape
            {"params": [self._scaling_base],  "lr": training_args.scaling_lr,           "name": "scaling"},
            {"params": [self._s],             "lr": training_args.scaling_lr,           "name": "axial"},
            # radial cage stored as (ρ̂, φ)
            {"params": [self._rho_hat],       "lr": 0.5 * training_args.scaling_lr,     "name": "rho"},
            {"params": [self._phi],           "lr": 0.5 * training_args.scaling_lr,     "name": "phi"},
            
            # Uncertainty parameters
            {"params": [self._gate_logit],    "lr": training_args.gate_lr,              "name": "gate"},
        ]
        
        if extra_parameters is not None:
            l += extra_parameters

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # cache β’s base LR so warm-up code can zero / restore it
        for pg in self.optimizer.param_groups:
            if pg["name"] == "beta":
                self._beta_base_lr = pg["lr"]

        # LR schedules for θ and β (others keep constant LR)
        self._theta_scheduler = get_expon_lr_func(
            lr_init  = training_args.theta_lr_init  * self.spatial_lr_scale,
            lr_final = training_args.theta_lr_final * self.spatial_lr_scale,
            lr_delay_mult = training_args.perm_lr_delay_mult,
            max_steps     = training_args.perm_lr_max_steps,
        )
        self._beta_scheduler = get_expon_lr_func(
            lr_init  = training_args.beta_lr_init   * self.spatial_lr_scale,
            lr_final = training_args.beta_lr_final  * self.spatial_lr_scale,
            lr_delay_mult = training_args.perm_lr_delay_mult,
            max_steps     = training_args.perm_lr_max_steps,
        )


    def update_learning_rate(self, iteration: int):
        """
        Apply per-step LR schedule to θ and β only.  All other groups keep
        their fixed learning-rate.
        """
        lr_theta = self._theta_scheduler(iteration)
        lr_beta  = self._beta_scheduler (iteration)

        for pg in self.optimizer.param_groups:
            if pg["name"] == "theta" and pg["lr"] > 0.0:
                pg["lr"] = lr_theta
            elif pg["name"] == "beta" and pg["lr"] > 0.0:
                pg["lr"] = lr_beta

    @staticmethod
    def _edge_dirs_to_quat(edge_dirs: torch.Tensor, eps=1e-6):
        """
        Convert direction vectors (M,3) to unit quaternions (M,4)
        that rotate +Y onto each direction.
        """
        up   = torch.tensor([0.0, 1.0, 0.0], device=edge_dirs.device)
        up   = up.expand_as(edge_dirs)
        v    = edge_dirs / (edge_dirs.norm(dim=1, keepdim=True) + eps)
        dot  = (up * v).sum(dim=1, keepdim=True).clamp(-1 + eps, 1 - eps)
        axis = torch.cross(up, v, dim=1)
        axis = axis / (axis.norm(dim=1, keepdim=True) + eps)
        ang  = torch.acos(dot)
        qw   = torch.cos(ang * 0.5)
        qxyz = axis * torch.sin(ang * 0.5)
        return torch.cat([qw, qxyz], dim=1)

 
    def _center_tangent(self, pts: torch.Tensor, s: torch.Tensor, eps: float = 1e-9):
        """
        Interpolate center positions and unit tangents along one strand.

        pts : (V, 3)  – strand vertices in metres
        s   : (K,)    – axial weights ∈ [0, 1]

        Returns
        -------
        center  : (K, 3)
        tangent : (K, 3)  – already L2-normalised
        """
        if s.numel() == 0:                             # nothing to do
            z = pts.new_zeros((0, 3))
            return z, z

        V   = pts.shape[0]
        t   = s * (V - 1)
        idx0 = torch.clamp(t.floor().long(), 0, V - 2)
        idx1 = idx0 + 1

        w = (t - idx0.float()).unsqueeze(-1)           # (K, 1)

        P0, P1 = pts[idx0], pts[idx1]
        center = (1.0 - w) * P0 + w * P1

        T_raw  = P1 - P0                               # (K, 3) *or* (3,)
        tangent = torch.nn.functional.normalize(T_raw, dim=-1, eps=eps)

        return center, tangent
    
    # ------------------------------------------------------------------
    def world_centers_and_tangents(self,
                                strand_vertices: torch.Tensor):
        """
        Compute the world-space center *and* tangent of every Gaussian
        from canonical strand vertices – without touching self._xyz.

        Parameters
        ----------
        strand_vertices : (S,V,3) tensor in metres (canonical pose).

        Returns
        -------
        centers  : (M,3) tensor
        tangents : (M,3) tensor  (unit length)
        """
        # use only strand Gaussians
        sid_all = self._strand_id           # (M_strand,)
        M_strand = sid_all.shape[0]
        if strand_vertices.dim() != 3:
            raise RuntimeError("strand_vertices must be (S,V,3)")
        if strand_vertices.shape[-1] != 3:                 # got (S,3,V)
            strand_vertices = strand_vertices.permute(0, 2, 1)

        S, V, _ = strand_vertices.shape

        s_lin = torch.sigmoid(self._s[sid_all])        # (M_strand,)
        t     = s_lin * (V - 1)                       # (M_strand,)
        idx0  = t.floor().long().clamp_max(V-2)       # left vertex
        idx1  = idx0 + 1                              # right vertex
        w     = (t - idx0.float()).unsqueeze(1)       # (M_strand,1)

        P0 = strand_vertices[sid_all, idx0]           # (M_strand,3)
        P1 = strand_vertices[sid_all, idx1]           # (M_strand,3)
        Tcur = P1 - P0                                # (M_strand,3)

        # ---- smooth tangent blend ---------------------------------------
        idx_prev = (idx0 - 1).clamp(min=0)
        idx_next = (idx1 + 1).clamp(max=V-1)
        Pprev = strand_vertices[sid_all, idx_prev]
        Pnext = strand_vertices[sid_all, idx_next]
        Tprev = P0 - Pprev
        Tnext = Pnext - P1
        Tprev[idx0 == 0]     = Tcur[idx0 == 0]
        Tnext[idx1 == V-1]   = Tcur[idx1 == V-1]
        blend0 = (1-w)*Tprev + w*Tcur
        blend1 = (1-w)*Tcur  + w*Tnext
        Traw   = (1-w)*blend0 + w*blend1
        tangent = torch.nn.functional.normalize(Traw, dim=1, eps=1e-9)

        # ---- radial offset ----------------------------------------------
        radii = self._strand_radius[sid_all].unsqueeze(1)         # (M_strand,1)
        rho   = torch.sigmoid(self._rho_hat[sid_all])             # (M_strand,1)
        phi   = self._phi[sid_all]                                # (M_strand,1)
        u = radii * rho * torch.cos(phi)
        v = radii * rho * torch.sin(phi)

        up  = tangent.new_tensor([0.,0.,1.]).expand_as(tangent)
        n1  = torch.cross(tangent, up, dim=1)
        bad = n1.norm(dim=1) < 1e-5
        if bad.any():
            alt = tangent.new_tensor([0.,1.,0.]).expand_as(tangent)
            n1[bad] = torch.cross(tangent[bad], alt[bad], dim=1)
        n1 = torch.nn.functional.normalize(n1, dim=1)
        n2 = torch.cross(tangent, n1, dim=1)

        centers_on_axis = (1-w)*P0 + w*P1
        centers = centers_on_axis + u*n1 + v*n2               # (M_strand,3)

        return centers, tangent

    @staticmethod
    @torch.no_grad()
    def _compute_curvature_torsion(p_neg1, p0, p1, p2, eps=1e-9):
        """
        Calculates curvature and torsion for a set of points on a polyline.
        
        Args:
            p_neg1, p0, p1, p2 (torch.Tensor): Tensors of shape (N, 3) representing four
                                              consecutive points on N polylines.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: kappa and tau, both of shape (N,).
        """
        # --- Curvature at p0 using points p_neg1, p0, p1 ---
        vec_0 = p0 - p_neg1
        vec_1 = p1 - p0
        vec_2 = p1 - p_neg1
        
        # Menger curvature formula
        num_k = 2.0 * torch.cross(vec_0, vec_1, dim=1).norm(dim=1)
        den_k = (vec_0.norm(dim=1) * vec_1.norm(dim=1) * vec_2.norm(dim=1)).clamp_min(eps)
        kappa = num_k / den_k

        # --- Torsion at p0/p1 via dihedral angle of planes (p_neg1,p0,p1) and (p0,p1,p2) ---
        tangent_01 = vec_1
        tangent_12 = p2 - p1
        
        # Binormals of the two planes
        binormal_0 = torch.cross(tangent_01, vec_0, dim=1)
        binormal_1 = torch.cross(tangent_12, tangent_01, dim=1)

        # Create a mask to handle degenerate (straight) cases where norm is zero
        b0_norm = binormal_0.norm(dim=1, keepdim=True)
        b1_norm = binormal_1.norm(dim=1, keepdim=True)
        safe_mask = ((b0_norm > eps) & (b1_norm > eps)).squeeze()
        
        # Initialize tau to zero for all points
        tau = torch.zeros_like(kappa)

        if safe_mask.any():
            # Only perform the calculation for non-degenerate cases
            b0_n = binormal_0[safe_mask] / b0_norm[safe_mask]
            b1_n = binormal_1[safe_mask] / b1_norm[safe_mask]

            # Angle between binormals
            x = torch.cross(b0_n, b1_n, dim=1)
            x_norm = x.norm(dim=1)
            y = (b0_n * b1_n).sum(dim=1).clamp(-1.0 + eps, 1.0 - eps)
            angle = torch.atan2(x_norm, y)

            # Sign of the torsion
            t1_safe = tangent_01[safe_mask]
            Tmid = t1_safe / t1_safe.norm(dim=1, keepdim=True).clamp_min(eps)
            sign = torch.sign((x * Tmid).sum(dim=1))
            
            # Normalize by local arc length
            arc = t1_safe.norm(dim=1).clamp_min(eps)
            
            # Update the tau tensor only for the safe indices
            tau[safe_mask] = sign * angle / arc
            
        return kappa, tau

    def update_xyz_rot_scale(self, strand_vertices, scalp_roots, *_):
        """
        Vectorised update of Gaussian centers, orientations and radial offsets, now
        with a smoothly-varying tangent (linear blend of neighbouring segments).

        Parameters
        ----------
        strand_vertices : (S, V, 3) tensor  – vertices of every strand, metres
        """
        # ------------------------------------------------------------------ input-shape sanity
        if strand_vertices.dim() != 3:
            raise RuntimeError("strand_vertices must be (S,V,3)")
        if strand_vertices.shape[-1] != 3:          # caller passed (S,3,V)
            strand_vertices = strand_vertices.permute(0, 2, 1)

        self.strands = strand_vertices
        S, V, _ = strand_vertices.shape

        sid  = self._strand_id
        M       = len(sid)

        # ------------------------------------------------------------------ strand-length cache
        seg          = (strand_vertices[:, 1:] - strand_vertices[:, :-1]).norm(dim=-1)  # (S,V-1)
        self.strand_length = seg.sum(dim=1)                                            # (S,)

        # ------------------------------------------------------------------ axial interpolation
        t    = self.get_axial_weight[:M] * (V - 1)          # (M,)
        idx0 = t.floor().long().clamp_max(V - 2)        # left  vertex
        idx1 = idx0 + 1                                 # right vertex
        w    = (t - idx0.float()).unsqueeze(1)          # (M,1)   weight in [0,1]

        P0   = strand_vertices[sid, idx0]               # (M,3)
        P1   = strand_vertices[sid, idx1]               # (M,3)
        Tcur = P1 - P0                                  # (M,3)   current segment dir

        # ------------------------------------------------------------------ smooth tangent (linear blend of neighbours)
        idx_prev = torch.clamp(idx0 - 1, 0)             # (M,)
        idx_next = torch.clamp(idx1 + 1, 0, V - 1)      # (M,)

        Pprev = strand_vertices[sid, idx_prev]
        Pnext = strand_vertices[sid, idx_next]

        Tprev = P0 - Pprev                              # (M,3)
        Tnext = Pnext - P1                              # (M,3)

        # fallback: if we copied the *same* vertex (idx0==0 or idx1==V-1) length is 0
        zero_prev = (idx0 == 0)
        zero_next = (idx1 == V - 1)
        if zero_prev.any():
            Tprev[zero_prev] = Tcur[zero_prev]
        if zero_next.any():
            Tnext[zero_next] = Tcur[zero_next]

        blend0  = (1.0 - w) * Tprev + w * Tcur
        blend1  = (1.0 - w) * Tcur  + w * Tnext
        Traw    = (1.0 - w) * blend0 + w * blend1           # (M,3)

        tangent = torch.nn.functional.normalize(Traw, dim=1, eps=1e-9)

        # ------------------------------------------------------------------ early-exit NaN / Inf guards
        if torch.isnan(tangent).any() or torch.isinf(tangent).any():
            print("WARNING: NaN/Inf in tangent – skipping update_xyz_rot_scale")
            return

        # ------------------------------------------------------------------ radial offsets (unchanged)
        radii = self._strand_radius[sid].unsqueeze(1)    # (M,1)
        rho   = self._activation(self._rho_hat[:M])          # (M,1)
        phi   = self._phi[:M]                                # (M,1)

        if torch.isnan(rho).any()  or torch.isinf(rho).any() \
        or torch.isnan(phi).any()  or torch.isinf(phi).any():
            print("WARNING: NaN/Inf in rho/phi – skipping update_xyz_rot_scale")
            return

        u = radii * rho * torch.cos(phi)                 # (M,1)
        v = radii * rho * torch.sin(phi)                 # (M,1)

        # ------------------------------------------------------------------ build local frame
        up  = tangent.new_tensor([0., 0., 1.]).expand_as(tangent)
        n1  = torch.cross(tangent, up, dim=1)
        bad = n1.norm(dim=1) < 1e-5                      # tangent ‖ up  ⇒  degenerate
        if bad.any():
            alt      = tangent.new_tensor([0., 1., 0.]).expand_as(tangent)
            n1[bad]  = torch.cross(tangent[bad], alt[bad], dim=1)
        n1 = torch.nn.functional.normalize(n1, dim=1)
        n2 = torch.cross(tangent, n1, dim=1)

        # ------------------------------------------------------------------ new centers & rotations
        centers = (1.0 - w) * P0 + w * P1                # linear interp. along strand
        xyz_new = centers + u * n1 + v * n2              # add radial offset

        self._xyz      = xyz_new
        self._rotation = self._edge_dirs_to_quat(tangent)

        # ------------------------------------------------------------------ keep legacy scaling update (optional)
        base = self._scaling_base[:M]
        perp = base[:, [0]]
        para = base[:, [1]]                              # let caller ensure σ‖ ≥ σ⊥
        self._scaling = torch.cat([perp, para, perp], dim=1)


        # ======================================================================
        #                  2.  SCALP‑DISK GAUSSIANS (indices = _scalp_id)
        # ======================================================================
        disk_idx = self._scalp_id
        if disk_idx.numel():
            # derive new root positions from first vertex of each strand
            root_pos = scalp_roots
            # inline normals computation (reuse neighbor_idx graph)
            k = 6
            # Compute neighbor indices using KD-tree on root_pos
            root_pos_unsq = root_pos.unsqueeze(0)  # (1, N, 3)
            _, idx, _ = knn_points(root_pos_unsq, root_pos_unsq, K=k+1, return_sorted=True)
            nbr_idx = idx[0, :, 1:]  # (N, k) — exclude self at idx[:, :, 0]

            nbrs    = root_pos[nbr_idx]                          # (S, k, 3)
            centerd = nbrs - root_pos.unsqueeze(1)               # (S, k, 3)
            cov     = centerd.transpose(2, 1).matmul(centerd) / float(k)  # (S, 3, 3)

            _, eigvec = torch.linalg.eigh(cov)
            normals_local = torch.nn.functional.normalize(eigvec[..., 0], dim=1)
            
            # offset disk centers to avoid Z-fighting
            centers = root_pos + normals_local * self.scalp_offset
            # orientations
            new_rot = self._edge_dirs_to_quat(normals_local)
            # scaling entries
            base = self._scaling_base[disk_idx]
            new_scale = base
            # append scalp rows to xyz, rotation, scaling
            self._xyz      = torch.cat([self._xyz,      centers], dim=0)
            self._rotation = torch.cat([self._rotation, new_rot],  dim=0)
            self._scaling  = torch.cat([self._scaling,  new_scale], dim=0)
            tangent = torch.cat([tangent, normals_local])

        # ======================================================================
        #  ----      REFACTORED CURVATURE & TORSION DIAGNOSTICS      ----
        # ======================================================================
        
        M_strands = self._strand_id.shape[0]
        S, V, _ = self.strands.shape
        sid = self._strand_id
        t_c = torch.sigmoid(self._s[:M_strands]) * (V - 1)
        h = 1.0
        eps = 1e-9

        # --- Step 1: Sample 5 points along the strand for each Gaussian ---
        def _interp_at(t):
            i0 = t.floor().long().clamp(0, V - 2)
            i1 = i0 + 1
            w  = (t - i0.float()).unsqueeze(1)
            P0 = self.strands[sid, i0]
            P1 = self.strands[sid, i1]
            return (1.0 - w) * P0 + w * P1

        # Sample points at t-2h, t-h, t, t+h, t+2h
        points = [_interp_at((t_c + (offset * h)).clamp(0.0, float(V - 1) - eps)) for offset in range(-2, 3)]
        
        # --- Step 2: Calculate curvature and torsion for the current and previous segments ---
        kappa, tau = self._compute_curvature_torsion(points[1], points[2], points[3], points[4], eps)
        kappa_prev, tau_prev = self._compute_curvature_torsion(points[0], points[1], points[2], points[3], eps)

        # --- Step 3: Calculate the gradients ---
        gkappa = (kappa - kappa_prev).abs()
        gtau   = (tau - tau_prev).abs()

        # --- Step 4: Cache the instantaneous values ---
        self.kappa  = kappa
        self.tau    = tau
        self.gkappa = gkappa
        self.gtau   = gtau

        return tangent

    @torch.no_grad()
    def compute_3D_filter(self, cameras, device="cuda"):
        #TODO consider focal length and image width
        xyz = self.get_xyz
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
        
        # we should use the focal length of the highest resolution camera
        focal_length = 0.
        for camera in cameras:
            old_device = camera.device if camera.device is not None else "cpu"
            camera.load2device(device)

            # transform points to camera space
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
             # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            xyz_cam = xyz @ R + T[None, :]
            
            xyz_to_cam = torch.norm(xyz_cam, dim=1)
            
            # project to screen space
            valid_depth = xyz_cam[:, 2] > 0.2
            
            
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            
            x = x / z * camera.focal_x + camera.image_width / 2.0
            y = y / z * camera.focal_y + camera.image_height / 2.0
            
            # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.image_width), torch.logical_and(y >= 0, y < camera.image_height))
            
            # use similar tangent space filtering as in the paper
            in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15), torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))
            
        
            valid = torch.logical_and(valid_depth, in_screen)
            
            # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < camera.focal_x:
                focal_length = camera.focal_x
            
            camera.load2device(old_device)
        
        distance[~valid_points] = distance[valid_points].max()
        
        #TODO remove hard coded value
        #TODO box to gaussian transform
        filter_3D = distance / focal_length * (0.2 ** 0.5)
        self.filter_3D = filter_3D[..., None]

    @torch.no_grad()
    def halve_large_parallel_sigmas(
        self,
        thick_perp: float = 5.645693247264717e-05,  # default σ⊥
        par_ratio:  float = 20.0,                  # default σ‖ / σ⊥
        thresh_mul_perp: float = 3.0                    # clamp when σ‖ ≥ 3× default
    ) -> int:
        """
        If σ‖ ≥ thick_perp * par_ratio * thresh_mul, divide that σ‖ by two.
        Returns the number of Gaussians affected.
        """

        # 1. Current sigmas for those rows
        sigma_perp = self._scaling_base[:len(self._strand_id), 0].exp()   # σ⊥   (N_scalp,)
        threshold_perp = thick_perp * thresh_mul_perp * self.global_scale

        # 2. Offending gaussians
        mask = (sigma_perp >= threshold_perp)
        n_fixed = int(mask.sum().item())
        if n_fixed == 0:
            return 0

        # 3. Halve both σ‖ and σ⊥ for masked rows → subtract ln 2 in log‑space
        idx_fix = torch.zeros(len(self._scaling_base)).bool()
        idx_fix[:len(self._strand_id)] = mask
        self._scaling_base.data[idx_fix, 0] -= math.log(thresh_mul_perp)   # σ⊥ /2  (cap size)

        return n_fixed

    # ================================================================
    #  ----  DENSIFICATION / PRUNE HELPERS  ---------------------------
    # ================================================================
    def _cat_to_optimizer(self, new_tensors: dict):
        """
        Concatenate rows from `new_tensors` onto each stored tensor, extending
        Adam moments where necessary.  Gradient accumulators and max_radii2D
        are kept in sync. Returns a dict of the *updated* tensors for easy re-binding.
        """
        out = {}
        added = 0  # how many new Gaussians were appended to xyz

        for name, ext in new_tensors.items():
            pg = next((g for g in self.optimizer.param_groups if g["name"] == name), None)
            if pg is not None:
                # — optimised tensor —
                old = pg["params"][0]
                state = self.optimizer.state.get(old, None)

                # extend Adam moments
                if state is not None:
                    z = torch.zeros_like(ext)
                    state["exp_avg"]    = torch.cat((state["exp_avg"],    z), 0)
                    state["exp_avg_sq"] = torch.cat((state["exp_avg_sq"], z), 0)

                # build new Parameter with old+ext
                new_p = nn.Parameter(torch.cat((old, ext), 0), requires_grad=True)
                if state is not None:
                    del self.optimizer.state[old]
                    self.optimizer.state[new_p] = state
                pg["params"][0] = new_p
                out[name] = new_p

            else:
                # — frozen tensor (xyz, rotation) —
                cur = getattr(self, f"_{name}")
                new = torch.cat((cur, ext), 0)
                setattr(self, f"_{name}", new)
                out[name] = new
                if name == "xyz":
                    added = ext.shape[0]

        # keep gradient statistics the same length as xyz
        if added > 0:
            z = torch.zeros((added, 1), device=self._s.device)
            self.xyz_gradient_accum = torch.cat((self.xyz_gradient_accum, z), 0)
            self.denom              = torch.cat((self.denom,              z), 0)
            # **also** extend max_radii2D buffer by zeros
            zeros_r = torch.zeros((added,), device=self.max_radii2D.device)
            self.max_radii2D = torch.cat((self.max_radii2D, zeros_r), 0)

        return out

    # ================================================================
    #  ----  PRUNE by opacity  ----------------------------------------
    # ================================================================
    @torch.no_grad()
    def prune_strands_by_opacity(self, opacity_thresh: float):
        """
        Remove any strand whose average Gaussian opacity (after 3D filtering)
        is less than `opacity_thresh`.  Drops both:
          • the strand’s root (from self.roots, strand_length, _strand_radius)
          • all Gaussians whose self._strand_id == that strand

        Keeps the optimizer’s state (exp_avg, exp_avg_sq) in sync.
        """
        # 1) compute per-strand mean opacity
        opac = self.get_opacity_with_3D_filter  # (M,)
        sid  = self._strand_id                              # (M,)
        S    = self.num_strands
        device = opac.device

        sum_op = torch.zeros(S, device=device)
        cnt    = torch.zeros(S, device=device)
        sum_op.index_add_(0, sid, opac)
        cnt   .index_add_(0, sid, torch.ones_like(opac))
        cnt = cnt.clamp(min=1.0)
        avg_op = sum_op / cnt                              # (S,)

        # 2) decide which strands to keep
        keep_strand = avg_op >= opacity_thresh
        if keep_strand.all():
            return  # nothing to prune

        # 3) build mapping old_strand → new_strand
        kept_idx = torch.nonzero(keep_strand, as_tuple=False).squeeze(1)
        new_ids = torch.zeros((S,), dtype=torch.long, device=device)
        new_ids[kept_idx] = torch.arange(len(kept_idx), device=device)

        # 4) mark Gaussians to keep
        keep_gauss = keep_strand[self._strand_id]           # (M,)

        # 5) prune optimizer-managed parameters
        new_params = self._prune_optimizer_rows(keep_gauss)
        # Re-bind each learnable from the returned dict:
        self._s             = new_params["axial"]
        self._rho_hat       = new_params["rho"]
        self._phi           = new_params["phi"]
        self._features_dc   = new_params["f_dc"]
        self._features_rest = new_params["f_rest"]
        self._features_asg  = new_params["f_asg"]
        self._opacity       = new_params["opacity"]
        self._scaling_base  = new_params["scaling"]

        # 6) prune frozen buffers (xyz, rotation, statistics)
        self._xyz               = self._xyz             [keep_gauss]
        self._rotation          = self._rotation        [keep_gauss]
        self.max_radii2D        = torch.zeros((self._xyz.shape[0]))
        self.xyz_gradient_accum = self.xyz_gradient_accum[keep_gauss]
        self.denom              = self.denom            [keep_gauss]

        # 7) rebuild _strand_id for the kept Gaussians
        old_sid = self._strand_id[keep_gauss]
        self._strand_id = new_ids[old_sid]

        # 8) rebuild _gauss_offset (prefix-sums of per-strand counts)
        S_new = keep_strand.sum().item()
        counts = torch.bincount(self._strand_id, minlength=S_new)
        offsets = torch.zeros((S_new+1,), dtype=torch.long, device=device)
        offsets[1:] = counts.cumsum(0)
        self._gauss_offset = offsets

        # 9) prune per-strand buffers (roots, radii, lengths)
        self.roots          = self.roots         [keep_strand]
        self.strand_length  = self.strand_length [keep_strand]
        self._strand_radius = self._strand_radius[keep_strand]
        self.num_strands    = S_new
        self.num_gaussians  = int(self._xyz.shape[0])

        # 10) recompute adjacency on the smaller root set
        self.compute_root_adjacency()

    def replace_tensor_to_optimizer(self, tensor: torch.Tensor, name: str):
        """
        Replace the parameter named `name` in Adam with a fresh tensor,
        preserving its optimizer-state (exp_avg, exp_avg_sq).
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                # grab and reattach optimizer buffers
                state = self.optimizer.state.pop(group["params"][0], {})
                state["exp_avg"]    = torch.zeros_like(tensor)
                state["exp_avg_sq"] = torch.zeros_like(tensor)

                # replace the Parameter
                new_p = nn.Parameter(tensor.clone().requires_grad_(True))
                group["params"][0] = new_p
                self.optimizer.state[new_p] = state

                optimizable_tensors[name] = new_p
        return optimizable_tensors

    def reset_opacity(self):
        """
        Clamp all current opacities to ≤0.01, then re-insert into optimizer
        so they’ll start optimizing again from that “low” value.
        """
        # inverse-sigmoid of min(current alpha, 0.01)
        alpha = self.get_opacity
        clipped = torch.min(alpha, torch.ones_like(alpha) * 0.01)
        new_logits = self.inverse_opacity_activation(clipped)

        # swap into optimizer
        tensors = self.replace_tensor_to_optimizer(new_logits, "opacity")
        self._opacity = tensors["opacity"]

    @torch.no_grad()
    def _prune_optimizer_rows(self, keep_mask: torch.BoolTensor):
        """
        Remove rows (dim0) where keep_mask==False from every per-Gaussian
        optimizer parameter (those whose first dim==M), re-attaching Adam state.
        Returns a dict name→new_parameter for re-binding on self.
        """
        out = {}
        for pg in self.optimizer.param_groups:
            name = pg["name"]
            old_p = pg["params"][0]
            # only prune if shape matches
            if old_p.dim() > 0 and old_p.shape[0] == keep_mask.shape[0]:
                kept = old_p[keep_mask]
                new_p = nn.Parameter(kept, requires_grad=True)
                # re-attach Adam stats
                state = self.optimizer.state.pop(old_p, None)
                if state is not None:
                    state["exp_avg"]    = state["exp_avg"][keep_mask]
                    state["exp_avg_sq"] = state["exp_avg_sq"][keep_mask]
                    self.optimizer.state[new_p] = state
                pg["params"][0] = new_p
                out[name] = new_p
            else:
                out[name] = old_p
        return out

    @torch.no_grad()
    def add_densification_stats(self,
                                viewspace_pts: torch.Tensor,
                                visible: torch.BoolTensor,
                                occ_mask: torch.BoolTensor,
                                outside_mask: torch.BoolTensor):
        """
        Accumulates screen-space gradients AND geometric diagnostics for visible
        Gaussians that are INSIDE the ground truth hair mask.
        """
        # 1. Get the global indices of all potentially visible Gaussians
        full_occ_indices = occ_mask.nonzero(as_tuple=False).squeeze(1)

        # 2. From that subset, find the ones that were actually rendered
        rendered_indices = full_occ_indices[visible]

        if rendered_indices.numel() == 0:
            return

        # --- FIX: Accumulate both the full 2D gradient and its norm ---
        # 3. Compute the full 2D gradients and their norms
        grad_xy_vec = viewspace_pts.grad[visible, :2] # Shape: (num_rendered, 2)
        grad_xy_norm = torch.norm(grad_xy_vec, dim=-1, keepdim=True) # Shape: (num_rendered, 1)

        # 4. Check which of these rendered Gaussians are inside the hair mask
        is_inside_mask = ~outside_mask[rendered_indices]
        
        # 5. Get the final set of indices and gradients to accumulate
        final_update_indices = rendered_indices[is_inside_mask]
        final_grad_xy_vec = grad_xy_vec[is_inside_mask]
        final_grad_xy_norm = grad_xy_norm[is_inside_mask]

        # 6. Accumulate the stats
        self.xyz_gradient_accum[final_update_indices] += final_grad_xy_norm
        self.xyz_full_grad_accum[final_update_indices] += final_grad_xy_vec
        self.denom[final_update_indices] += 1
        
        # --- Geometric Diagnostic Accumulation ---
        M_strands = self._strand_id.shape[0]
        gkappa = self.gkappa[:M_strands]
        gtau = self.gtau[:M_strands]
        
        inside_mask_float = (~outside_mask[:M_strands]).float().unsqueeze(1)
        
        gk_safe = torch.where(torch.isfinite(gkappa), gkappa, 0.0).detach()
        gt_safe = torch.where(torch.isfinite(gtau),   gtau,   0.0).detach()
        
        self.strand_grad_kappa_accum.add_((gk_safe.unsqueeze(1)) * inside_mask_float)
        self.strand_grad_tau_accum.add_((gt_safe.unsqueeze(1)) * inside_mask_float)
        self.strand_grad_denom.add_(inside_mask_float)

    @torch.no_grad()
    def densify_and_prune(self,
                          radii,
                          grad_thresh,
                          min_opac = 0.005,
                          th_gkappa = 500.0 * 20,
                          th_gtau = 500.0 * 20,
                          th_merge_k = 0.0394 * 20,
                          th_merge_t = 1.9508 * 20,
                          th_overlap = 0.006,
                          th_appearance = 999999999,
                          th_grad_variance = 0.045
                          ):
        """
        Performs one step of adaptive density control using a hybrid strategy.
        This version uses an atomic update and resets accumulators after the step.
        """
        M0 = self._strand_id.shape[0]
        dev = self._xyz.device

        import pdb; pdb.set_trace()

        # 0) Aggregate running diagnostics from the period since the last densification
        denom = self.strand_grad_denom[:M0].clamp(min=1e-9)
        avg_gkappa = (self.strand_grad_kappa_accum[:M0] / denom).squeeze(-1)
        avg_gtau = (self.strand_grad_tau_accum[:M0] / denom).squeeze(-1)
        avg_gkappa[torch.isnan(avg_gkappa)] = 0.0
        avg_gtau[torch.isnan(avg_gtau)] = 0.0

        # 1. IDENTIFY PROBLEMATIC GAUSSIANS
        denom_grad = self.denom[:M0].clamp(min=1e-9)
        avg_grad_norm = self.xyz_gradient_accum[:M0] / denom_grad
        problematic_mask = (avg_grad_norm.squeeze(-1) >= grad_thresh) & (radii[:M0] > 0)
        
        # # Suppress densification for strands that are just moving (gradient coherence check)
        # avg_full_grad = self.xyz_full_grad_accum[:M0] / denom_grad

        # strand_grad_variance = torch.zeros(self.num_strands, device=dev)
        # mean_grad_sq = torch.zeros(self.num_strands, 2, device=dev)
        # mean_grad = torch.zeros(self.num_strands, 2, device=dev)
        # strand_counts = torch.zeros(self.num_strands, 1, device=dev).clamp_min(1e-9)

        # strand_ids_of_problematic = self._strand_id[problematic_mask]
        
        # if strand_ids_of_problematic.numel() > 0:
        #     mean_grad_sq.scatter_add_(0, strand_ids_of_problematic.unsqueeze(1).expand(-1, 2), avg_full_grad[problematic_mask]**2)
        #     mean_grad.scatter_add_(0, strand_ids_of_problematic.unsqueeze(1).expand(-1, 2), avg_full_grad[problematic_mask])
        #     strand_counts.scatter_add_(0, strand_ids_of_problematic.unsqueeze(1), torch.ones_like(strand_ids_of_problematic, dtype=torch.float32).unsqueeze(1))
            
        #     strand_mean_grad = mean_grad / strand_counts
        #     strand_mean_grad_sq = mean_grad_sq / strand_counts
        #     strand_grad_variance = (strand_mean_grad_sq - strand_mean_grad**2).sum(dim=1)

        # coherent_motion_strands = strand_grad_variance < th_grad_variance
        # suppress_densification_mask = coherent_motion_strands[self._strand_id]
        # problematic_mask &= ~suppress_densification_mask
        
        # 2. CREATE MASKS FOR ALL ACTIONS
        is_geom_complex = (avg_gkappa > th_gkappa) | (avg_gtau > th_gtau)
        split_mask = problematic_mask & is_geom_complex
        # clone_mask = problematic_mask & (~is_geom_complex)

        opacity_prune_mask = (self.get_opacity[:M0] < min_opac).squeeze()
        merge_pairs = self._find_merge_pairs(avg_gkappa, avg_gtau, th_merge_k, th_merge_t, th_overlap, th_appearance)
        
        # 3. CONSOLIDATE AND EXECUTE
        merge_victims_indices = torch.tensor([pair[1] for pair in merge_pairs], device=dev, dtype=torch.long) if merge_pairs else torch.tensor([], device=dev, dtype=torch.long)
        
        combined_prune_mask = torch.zeros(M0, dtype=torch.bool, device=dev)
        combined_prune_mask[merge_victims_indices] = True
        combined_prune_mask |= opacity_prune_mask
        combined_prune_mask |= split_mask

        # clone_mask[combined_prune_mask] = False
        # import pdb; pdb.set_trace()

        
        n_merge = self._merge(merge_pairs)
        split_children_tensors = self._get_split_children(split_mask)
        # clone_children_tensors = self._get_clone_children(clone_mask)
        
        full_prune_mask = torch.zeros(self.num_gaussians, dtype=torch.bool, device=dev)
        full_prune_mask[:M0] = combined_prune_mask
        n_prune_total = self._prune(full_prune_mask)
        
        n_split = self._add_new_gaussians(split_children_tensors)
        # n_clone = self._add_new_gaussians(clone_children_tensors)

        # 4) Final bookkeeping: Reset accumulators for the next densification window
        M_new = int(self._s.shape[0])
        self.num_gaussians = M_new

        N_scalp = len(self.scalp_roots)
        self.xyz_gradient_accum = torch.zeros((M_new, 1), device=dev)
        self.xyz_full_grad_accum = torch.zeros((M_new, 2), device=dev)
        self.denom = torch.zeros((M_new, 1), device=dev)
        self.strand_grad_kappa_accum = torch.zeros((M_new - N_scalp, 1), device=dev)
        self.strand_grad_tau_accum   = torch.zeros((M_new - N_scalp, 1), device=dev)
        self.strand_grad_denom       = torch.zeros((M_new - N_scalp, 1), device=dev)

        return n_split, 0, n_merge, n_prune_total

    @torch.no_grad()
    def _find_merge_pairs(self, avg_gkappa, avg_gtau, th_merge_k, th_merge_t, th_overlap, th_appearance):
        """Helper to vectorize the search for mergeable pairs."""
        
        # --- FIX: Explicitly work only on the strand Gaussians ---
        M_strands = self._strand_id.shape[0]
        
        # Slice all attributes to match the size of the input diagnostics (avg_gkappa)
        strand_id = self._strand_id
        s_values = torch.sigmoid(self._s[:M_strands])
        xyz = self._xyz[:M_strands]
        scaling = self.get_scaling[:M_strands]
        features = self.get_features[:M_strands]
        # --- END FIX ---
        
        is_geom_simple = (avg_gkappa < th_merge_k) & (avg_gtau < th_merge_t)
        
        sort_key = strand_id.float() + s_values
        sorted_global_indices = torch.argsort(sort_key)
        
        g_idx1_all = sorted_global_indices[:-1]
        g_idx2_all = sorted_global_indices[1:]
        
        valid_pair_mask = (strand_id[g_idx1_all] == strand_id[g_idx2_all])
        g_idx1 = g_idx1_all[valid_pair_mask]
        g_idx2 = g_idx2_all[valid_pair_mask]

        if g_idx1.numel() == 0:
            return []

        geom_simple_check = is_geom_simple[g_idx1] & is_geom_simple[g_idx2]
        
        dist = torch.norm(xyz[g_idx1] - xyz[g_idx2], dim=1)
        sigma_max1 = scaling[g_idx1].max(dim=1).values
        sigma_max2 = scaling[g_idx2].max(dim=1).values
        overlap_check = (dist <= th_overlap * self.global_scale * (sigma_max1 + sigma_max2))
        
        if self.uniform_strand_color:
            appearance_check = torch.ones_like(g_idx1, dtype=torch.bool)
        else:
            features1 = features[g_idx1].clamp(-1, 1)
            features2 = features[g_idx2].clamp(-1, 1)
            feat_dist = torch.norm(features1 - features2, dim=(1, 2))
            appearance_check = (feat_dist <= th_appearance)

        final_merge_mask = geom_simple_check & overlap_check & appearance_check
        merge_indices_1 = g_idx1[final_merge_mask]
        merge_indices_2 = g_idx2[final_merge_mask]

        if merge_indices_1.numel() > 0:
            return list(zip(merge_indices_1.tolist(), merge_indices_2.tolist()))
        return []

    @torch.no_grad()
    def _get_clone_children(self, clone_mask):
        """
        Gathers attributes for Gaussians to be cloned, correctly handling strand-only indexing.
        """
        if not clone_mask.any():
            return None

        M_strands = self._strand_id.shape[0]
        
        # Dictionary for learnable parameters, using optimizer names as keys
        params_to_clone = {
            "axial": self._s.data[:M_strands][clone_mask],
            "rho": self._rho_hat.data[:M_strands][clone_mask],
            "phi": self._phi.data[:M_strands][clone_mask],
            "scaling": self._scaling_base.data[:M_strands][clone_mask],
            "gate": self._gate_logit.data[:M_strands][clone_mask],
        }
        if not self.uniform_strand_color:
            params_to_clone.update({
                "f_dc": self._features_dc.data[:M_strands][clone_mask],
                "f_rest": self._features_rest.data[:M_strands][clone_mask],
                "f_asg": self._features_asg.data[:M_strands][clone_mask],
                "opacity": self._opacity.data[:M_strands][clone_mask]
            })

        # Dictionary for non-parameter buffers
        buffers_to_clone = {
            "xyz": self._xyz[:M_strands][clone_mask],
            "rotation": self._rotation[:M_strands][clone_mask],
            "strand_id": self._strand_id[clone_mask],
        }
        
        return {**params_to_clone, **buffers_to_clone}

    @torch.no_grad()
    def _get_split_children(self, split_mask):
        """
        Calculates attributes for the new children of split Gaussians, 
        correctly handling strand-only indexing.
        """
        if not split_mask.any():
            return None
        
        M_strands = self._strand_id.shape[0]
        
        parents_s = self._s.data[:M_strands][split_mask]
        parents_scaling = self._scaling_base.data[:M_strands][split_mask]
        parents_strand_id = self._strand_id[split_mask]
        
        child_scaling = parents_scaling.clone()
        child_scaling[:, 1] -= math.log(1.6)
        
        strand_lengths = self.strand_length[parents_strand_id]
        delta_s = (torch.exp(child_scaling[:, 1]) / (strand_lengths + 1e-9)).clamp_max(0.5)
        
        s_lin_parents = torch.sigmoid(parents_s)
        s_lin_child1 = (s_lin_parents + delta_s).clamp(0.0001, 0.9999)
        s_lin_child2 = (s_lin_parents - delta_s).clamp(0.0001, 0.9999)
        s_child1 = inverse_sigmoid(s_lin_child1)
        s_child2 = inverse_sigmoid(s_lin_child2)

        def dup(x):
            # Slices the tensor to only include strands before indexing
            return x[:M_strands][split_mask].repeat(2, *[1] * (x.dim() - 1))

        # Dictionary for learnable parameters, using optimizer names as keys
        params_to_split = {
            "axial": torch.cat([s_child1, s_child2]),
            "rho": dup(self._rho_hat.data),
            "phi": dup(self._phi.data),
            "scaling": child_scaling.repeat(2, 1),
            "gate": dup(self._gate_logit.data),
        }
        if not self.uniform_strand_color:
            params_to_split.update({
                "f_dc": dup(self._features_dc.data),
                "f_rest": dup(self._features_rest.data),
                "f_asg": dup(self._features_asg.data),
                "opacity": dup(self._opacity.data),
            })
            
        # Dictionary for non-parameter buffers
        buffers_to_split = {
            "xyz": dup(self._xyz),
            "rotation": dup(self._rotation),
            "strand_id": parents_strand_id.repeat(2),
        }

        return {**params_to_split, **buffers_to_split}

    @torch.no_grad()
    def _prune(self, prune_mask):
        """
        Performs a consolidated prune of all flagged Gaussians.
        The input mask must be the size of ALL Gaussians (M_total).
        """
        if not prune_mask.any():
            return 0
        
        device = self._xyz.device
        
        keep_mask = ~prune_mask
        n_pruned = prune_mask.sum().item()
        
        # This helper function already correctly handles pruning full tensors
        param_name_map = {
            "axial": "_s", "rho": "_rho_hat", "phi": "_phi",
            "scaling": "_scaling_base", "gate": "_gate_logit",
            "f_dc": "_features_dc", "f_rest": "_features_rest",
            "f_asg": "_features_asg", "opacity": "_opacity"
        }

        # This helper correctly prunes the optimizer state and returns a dict
        # keyed by the optimizer param_group name (e.g., "axial").
        new_params = self._prune_optimizer_rows(keep_mask)

        # Re-bind all parameters using the correct attribute name from the map
        for name, new_param in new_params.items():
            if name in param_name_map:
                attr_name = param_name_map[name]
                setattr(self, attr_name, new_param)

        # Prune non-parameter buffers and accumulators
        self._xyz = self._xyz[keep_mask]
        self._rotation = self._rotation[keep_mask]
        self.max_radii2D = self.max_radii2D[keep_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[keep_mask]
        self.xyz_full_grad_accum = self.xyz_full_grad_accum[keep_mask]
        self.denom = self.denom[keep_mask]
        
        # Strand-specific buffers need careful handling if strands are removed
        self._strand_id = self._strand_id[keep_mask[:len(self._strand_id)]]
        
        if self.uniform_strand_color or n_pruned > 0 : # Rebuild if any strand might have been removed
            strands_to_keep = torch.unique(self._strand_id)
            if strands_to_keep.numel() < self.num_strands:
                strand_keep_mask = torch.zeros(self.num_strands, dtype=torch.bool, device=device)
                strand_keep_mask[strands_to_keep] = True
                
                remap_table = -torch.ones(self.num_strands, dtype=torch.long, device=device)
                remap_table[strands_to_keep] = torch.arange(strands_to_keep.shape[0], device=device)
                self._strand_id = remap_table[self._strand_id]
                
                self._strand_radius = self._strand_radius[strand_keep_mask]
                self.strand_length = self.strand_length[strand_keep_mask]
                self.num_strands = int(strands_to_keep.shape[0])
        
        # Rebuild offset table and update counts
        if self.num_strands > 0:
            counts = torch.bincount(self._strand_id, minlength=self.num_strands)
            self._gauss_offset = torch.cat([torch.tensor([0], device=device, dtype=torch.long), counts.cumsum(0)])
        else:
            self._gauss_offset = torch.tensor([0], device=device, dtype=torch.long)

        M_strands_after = self._strand_id.shape[0]
        self._scalp_id = torch.arange(M_strands_after, M_strands_after + self._scalp_id.shape[0], device=device)
        self.num_gaussians = self._s.shape[0]
        return n_pruned

    @torch.no_grad()
    def _add_new_gaussians(self, new_tensors_dict):
        """
        Consolidated function to add new Gaussians, inserting them before the scalp block.
        """
        if new_tensors_dict is None or not new_tensors_dict:
            return 0

        device = self._xyz.device
        
        n_added = new_tensors_dict["xyz"].shape[0]
        M_strands_before = self._strand_id.shape[0]
        M_total_before = self.num_gaussians

        # THIS MAP IS THE CRUCIAL FIX
        param_name_map = {
            "axial": "_s", "rho": "_rho_hat", "phi": "_phi",
            "scaling": "_scaling_base", "gate": "_gate_logit",
            "f_dc": "_features_dc", "f_rest": "_features_rest",
            "f_asg": "_features_asg", "opacity": "_opacity"
        }
        params_to_add = {name: data for name, data in new_tensors_dict.items() if name in param_name_map}

        # 1. Append parameter data and update optimizer state
        updated_params = self._cat_to_optimizer(params_to_add)
        for name, new_param in updated_params.items():
            setattr(self, param_name_map[name], new_param)

        # 2. Append buffer and accumulator data
        self._xyz = torch.cat([self._xyz, new_tensors_dict["xyz"]], dim=0)
        self._rotation = torch.cat([self._rotation, new_tensors_dict["rotation"]], dim=0)
        self.max_radii2D = torch.cat([self.max_radii2D, torch.zeros(n_added, device=device)], dim=0)
        
        # 3. Create reordering indices
        old_strand_indices = torch.arange(M_strands_before, device=device)
        new_strand_indices = torch.arange(M_total_before, M_total_before + n_added, device=device)
        scalp_indices = torch.arange(M_strands_before, M_total_before, device=device)
        reorder_indices = torch.cat([old_strand_indices, new_strand_indices, scalp_indices])

        # 4. Apply reordering to all per-Gaussian tensors
        M_total_new = M_total_before + n_added
        for pg in self.optimizer.param_groups:
            param = pg["params"][0]
            if param.dim() > 0 and param.shape[0] == M_total_new:
                state = self.optimizer.state.get(param)
                param.data = param.data[reorder_indices].contiguous()
                if state:
                    state['exp_avg'] = state['exp_avg'][reorder_indices].contiguous()
                    state['exp_avg_sq'] = state['exp_avg_sq'][reorder_indices].contiguous()
        
        # 5. Update lookup tables and counts
        self._strand_id = torch.cat([self._strand_id, new_tensors_dict["strand_id"]], dim=0)
        M_strands_after = self._strand_id.shape[0]
        self._scalp_id = torch.arange(M_strands_after, M_strands_after + self._scalp_id.shape[0], device=device)
        self.num_gaussians = self._s.shape[0]
        
        counts = torch.bincount(self._strand_id, minlength=self.num_strands)
        self._gauss_offset = torch.cat([torch.tensor([0], device=device, dtype=torch.long), counts.cumsum(0)])

        return n_added

    @torch.no_grad()
    def _merge(self, merge_pairs):
        """
        Updates survivor attributes by merging victim attributes into them.
        This function DOES NOT remove the victims; it only performs the update.
        """
        if not merge_pairs:
            return 0

        for survivor_idx, victim_idx in merge_pairs:
            # --- 1. Update Geometric Properties (Always Per-Gaussian) ---
            
            # Lenient merge for radial position (pull towards core)
            rho_surv = torch.sigmoid(self._rho_hat.data[survivor_idx])
            rho_vict = torch.sigmoid(self._rho_hat.data[victim_idx])
            self._rho_hat.data[survivor_idx] = inverse_sigmoid(torch.min(rho_surv, rho_vict))
            
            # Average axial position and azimuth
            self._s.data[survivor_idx] = (self._s.data[survivor_idx] + self._s.data[victim_idx]) * 0.5
            self._phi.data[survivor_idx] = (self._phi.data[survivor_idx] + self._phi.data[victim_idx]) * 0.5

            # Survivor's scale should encompass the victim's
            surv_scale_par = torch.exp(self._scaling_base.data[survivor_idx, 1])
            vict_scale_par = torch.exp(self._scaling_base.data[victim_idx, 1])
            self._scaling_base.data[survivor_idx, 1] = torch.log(torch.max(surv_scale_par, vict_scale_par))
            
            # --- 2. Update Appearance Properties (Only in Per-Gaussian Mode) ---
            
            # If color is per-strand, merging two Gaussians on that same strand
            # should not change the strand's single color. Therefore, we skip the update.
            if not self.uniform_strand_color:
                self._features_dc.data[survivor_idx] = (self._features_dc.data[survivor_idx] + self._features_dc.data[victim_idx]) * 0.5
                self._features_rest.data[survivor_idx] = (self._features_rest.data[survivor_idx] + self._features_rest.data[victim_idx]) * 0.5
                self._opacity.data[survivor_idx] = (self._opacity.data[survivor_idx] + self._opacity.data[victim_idx]) * 0.5
                self._features_asg.data[survivor_idx] = (self._features_asg.data[survivor_idx] + self._features_asg.data[victim_idx]) * 0.5
                self._gate_logit.data[survivor_idx] = (self._gate_logit.data[survivor_idx] + self._gate_logit.data[victim_idx]) * 0.5
                
        return len(merge_pairs)