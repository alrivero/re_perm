import torch
import trimesh
import pickle
import math
import numpy as np
import torch.nn as nn
import scipy.spatial as _sp
import scipy

from random import random
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, strip_symmetric, build_scaling_rotation, quatProduct_batch

STRAND_VERTEX_COUNT = 100
SCALE_DIVISOR = 1

class GaussianPerm(nn.Module):
    def __init__(self, perm, pseudo_roots,
                 start_hair_style, sh_degree, asg_degree,
                 num_strands=10000,
                 neighbor_k=6,
                 num_gaussians=2000000,
                 cached_roots=None):
        super().__init__()

        self.perm          = perm
        self.num_strands   = num_strands
        self.max_sh_degree = sh_degree
        self.max_asg_degree = asg_degree
        self.neighbor_k    = neighbor_k
        self.uniform_strand_color = False

        # sample scalp roots once
        self.roots, self.global_strand_radii = self.perm.hair_roots.sample_scalp_hex(
            num_strands, pseudo_roots
        )
        self.roots = self.roots.to(self.perm.device)

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

    def init_parameters(
        self,
        start_hair_style,
        total_gaussians: int = 2000000,
        thick_perp: float = 5.645693247264717e-05,
        par_ratio: float = 20,
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
        strand_R = torch.ones((self.num_strands,), device=device) * (self.global_strand_radii)
        self.register_buffer("_strand_radius", strand_R, persistent=False)

        # ── 3) Gradient accumulators & 2D‐radii ────────────────────────────
        self.xyz_gradient_accum = torch.zeros((M,1), device=device)
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
        self._features_dc   = nn.Parameter(feats[:,:,:1].transpose(1,2).contiguous(),
                                            requires_grad=True)
        self._features_rest = nn.Parameter(feats[:,:,1:].transpose(1,2).contiguous(),
                                            requires_grad=True)
        self._opacity       = nn.Parameter(self.inverse_opacity_activation(
                                            torch.ones((M,1), device=device)),
                                        requires_grad=True)

        # ── 6) Default scale bases ──────────────────────────────────────────
        sigma_perp0 = thick_perp 
        sigma_par0  = sigma_perp0 * par_ratio
        log0        = torch.tensor([sigma_perp0, sigma_par0, sigma_perp0],
                                device=device).log()
        self._scaling_base = nn.Parameter(log0.expand(M,3).clone(), requires_grad=True)
        self._scaling      = None

        # ── 7) ASG Initialization ─────────-────────────────────────────────
        self._features_asg = nn.Parameter(torch.zeros(M, self.max_asg_degree)).float()

        # ── 8) Optional duplication ─────────────────────────────────────────
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
            self.denom              = torch.zeros_like(self.xyz_gradient_accum)
            self.register_buffer("max_radii2D",
                                torch.zeros((M,), device=device),
                                persistent=False)

        self.active_sh_degree = 0
        self.spatial_lr_scale = 1.0
        self.compute_root_adjacency()
    
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
        total_gaussians: int = 3_000_000,
        thick_perp: float = 5.645693247264717e-05,
        par_ratio:   float = 20.0,
        k_neighbors: int   = 2,
        dup_factor:  int   = 1,
        uniform_strand_color: bool = False    # ← NEW FLAG
    ):
        """
        Re-initialise the Gaussian scene on a new set of scalp roots.

        Parameters
        ----------
        uniform_strand_color
            If **True**, colour SH coefficients, ASG features and opacity logits
            are stored *once per strand* (shape `(S_new, …)`); the usual attribute
            names (`_features_dc`, `_features_rest`, `_features_asg`, `_opacity`)
            are **re-used**.  The properties `get_features`, `get_asg_features`
            and `get_opacity` automatically broadcast them to `(M, …)` when
            accessed, so no other code must change.
        """
        device = self._xyz.device
        dtype  = self._xyz.dtype

        # ------------------------------------------------------------------ 0.  flatten existing features
        M_old = self._s.shape[0]

        dc_shape   = self._features_dc.shape[1:]        # (1, n_dc)
        rest_shape = self._features_rest.shape[1:]      # (3, n_rest)
        Ddc  = int(np.prod(dc_shape))                   # scalar
        Dr   = int(np.prod(rest_shape))
        Dasg = self._features_asg.shape[1]

        old_dc_flat   = self._features_dc.reshape(M_old, Ddc)
        old_rest_flat = self._features_rest.reshape(M_old, Dr)
        old_asg_flat  = self._features_asg                 # (M_old, Dasg)

        # ------------------------------------------------------------------ 1.  per-old-strand means
        S_old = self.num_strands
        strand_feats_colour = torch.zeros((S_old, Ddc + Dr), device=device, dtype=dtype)
        strand_feats_asg    = torch.zeros((S_old, Dasg     ), device=device, dtype=dtype)
        counts              = torch.zeros((S_old,),        device=device, dtype=dtype)

        strand_feats_colour.index_add_(0, self._strand_id,
                                    torch.cat([old_dc_flat, old_rest_flat], dim=1))
        strand_feats_asg   .index_add_(0, self._strand_id, old_asg_flat)
        counts.index_add_(0, self._strand_id,
                        torch.ones_like(self._strand_id, dtype=dtype))
        nz = counts > 0
        strand_feats_colour[nz] /= counts[nz].unsqueeze(1)
        strand_feats_asg   [nz] /= counts[nz].unsqueeze(1)

        # ------------------------------------------------------------------ 2.  k-NN colour transfer to new roots
        old_np = self.roots.cpu().numpy()
        new_np = new_roots.cpu().numpy()
        tree   = _sp.cKDTree(old_np)
        _, nearest_neigh = tree.query(new_np, k=k_neighbors)    # (S_new, k)

        S_new = new_roots.shape[0]
        new_colour_per_strand = torch.zeros((S_new, Ddc + Dr), device=device, dtype=dtype)
        new_asg_per_strand    = torch.zeros((S_new, Dasg     ), device=device, dtype=dtype)

        for i in range(S_new):
            neigh = [j for j in nearest_neigh[i] if j < S_old]
            if neigh:
                new_colour_per_strand[i] = strand_feats_colour[neigh].mean(0)
                new_asg_per_strand   [i] = strand_feats_asg   [neigh].mean(0)
            else:                                  # degenerate – randomise
                new_colour_per_strand[i].uniform_(-1, 1)
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
        strand_R = torch.ones((S_new,), device=device, dtype=dtype) * new_radii
        self.register_buffer("_strand_radius", strand_R, persistent=False)

        # ------------------------------------------------------------------ 7.  centres & quats
        centers, quats = [], []
        for sid in range(S_new):
            a, b = offsets[sid].item(), offsets[sid+1].item()
            if b > a:
                ss = torch.linspace(0, 1, b - a, device=device, dtype=dtype)
                c, t = self._center_tangent_normal(strands[sid], ss)
                centers.append(c);  quats.append(self._edge_dirs_to_quat(t))
        self._xyz      = torch.cat(centers, 0)
        self._rotation = torch.cat(quats,   0)

        # ------------------------------------------------------------------ 8.  initialise colour / opacity / ASG
        if uniform_strand_color:
            # -------- one vector per strand (shape (S_new, …)) ---------------
            self._features_dc   = nn.Parameter(
                new_colour_per_strand[:, :Ddc].view(S_new, 1, Ddc),
                requires_grad=True
            )
            self._features_rest = nn.Parameter(
                new_colour_per_strand[:, Ddc:].view(S_new, 3, Dr // 3),
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
            colour = new_colour_per_strand[self._strand_id]   # (M_new, Ddc+Dr)
            asg    = new_asg_per_strand   [self._strand_id]   # (M_new, Dasg)

            dc_flat = colour[:, :Ddc]
            rs_flat = colour[:, Ddc:]

            self._features_dc   = nn.Parameter(
                dc_flat.view(self.num_gaussians, 1, Ddc).transpose(1, 2).contiguous(),
                requires_grad=True
            )
            self._features_rest = nn.Parameter(
                rs_flat.view(self.num_gaussians, 3, Dr // 3).transpose(1, 2).contiguous(),
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
        self._phi      = nn.Parameter(
            torch.empty((self.num_gaussians, 1), device=device, dtype=dtype)
                .uniform_(0, 2*math.pi),
            requires_grad=True
        )
        rho            = torch.rand((self.num_gaussians, 1), device=device, dtype=dtype)
        self._rho_hat  = nn.Parameter(self.inverse_rho_activation(rho), True)

        sigma_p = thick_perp
        sigma_l = sigma_p * par_ratio
        log0    = torch.tensor([sigma_p, sigma_l, sigma_p],
                            device=device, dtype=dtype).log()
        self._scaling_base = nn.Parameter(
            log0.expand(self.num_gaussians, 3).clone(), True
        )
        self._scaling = None

        # ------------------------------------------------------------------ 10.  accumulators
        self.xyz_gradient_accum = torch.zeros((self.num_gaussians, 1), device=device, dtype=dtype)
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
            self.denom              = torch.zeros_like(self.xyz_gradient_accum)
            self.register_buffer("max_radii2D",
                                torch.zeros((self.num_gaussians,), device=device, dtype=dtype),
                                persistent=False)

        # ------------------------------------------------------------------ 12.  internal bookkeeping
        self.roots       = new_roots.to(device, dtype=dtype)
        self.num_strands = S_new
        self.compute_root_adjacency()

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
        self.global_strand_radii = new_radii
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
        Pack everything required to resume training later,
        including per-strand Voronoi radii.
        """
        return (
            # 0–1: train-time state
            self.active_sh_degree,
            self.spatial_lr_scale,

            # 2–8: learnable per-Gaussian tensors
            self._s,
            self._rho_hat,
            self._phi,
            self._features_dc,
            self._features_rest,
            self._features_asg,
            self._opacity,
            self._scaling_base,

            # 9: optimizer state
            self.optimizer.state_dict(),

            # 10–13: PERM latents + roots
            self.theta,
            self.beta,
            self.roots,

            # 14–15: lookup buffers
            self._strand_id,
            self._gauss_offset,

            # 16: per-strand cage radii
            self._strand_radius,     # (S,)

            # 17–18: gradient statistics
            self.xyz_gradient_accum,
            self.denom,
            self.max_radii2D,
        )

    def restore(self, model_args, training_args, extra_parameters=None):
        """
        Unpack everything from capture(), rebuild optimizer, reload state,
        and re-register all buffers.
        """
        (
            self.active_sh_degree,
            self.spatial_lr_scale,

            self._s,
            self._rho_hat,
            self._phi,
            self._features_dc,
            self._features_rest,
            self._features_asg,
            self._opacity,
            self._scaling_base,

            opt_state_dict,

            self.theta,
            self.beta,
            self.roots,

            strand_id_buf,
            gauss_offset_buf,
            strand_radius_buf,

            xyz_grad_accum,
            denom,
            self.max_radii2D,
        ) = model_args

        # 1) re-register those three as buffers so they move with .to(device):
        self.register_buffer("_strand_id",    strand_id_buf)
        self.register_buffer("_gauss_offset", gauss_offset_buf)
        self.register_buffer("_strand_radius", strand_radius_buf)

        # 2) restore gradient stats
        self.xyz_gradient_accum = xyz_grad_accum
        self.denom              = denom

        # 3) rebuild optimizer (this sees all your nn.Parameter fields)
        self.training_setup(training_args, extra_parameters)
        # 4) load its saved state
        self.optimizer.load_state_dict(opt_state_dict)

        # 5) recompute any cached / derived state
        self.num_gaussians = self._s.shape[0]
        self.num_strands   = self.roots.shape[0]
        self.compute_root_adjacency()

        # if you rely on self._xyz / self._rotation being fresh,
        # call update_xyz_rot_scale(...) here with whatever your
        # canonical strand vertices are.

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

        perp = raw[:, 0] # torch.minimum(raw[:, 0], self._strand_radius.mean()) # FIX ME
        paral = torch.maximum(raw[:, 1], perp * 1.5)  # clamp sigma‖
        return torch.stack((perp, paral, perp), dim=1)

    @property
    def get_features(self):
        if getattr(self, "uniform_strand_color", False):
            # ─ strand-level tensors → per-Gaussian via self._strand_id
            dc   = self._features_dc  [self._strand_id]          # (M,1,n_dc)
            rest = self._features_rest[self._strand_id]          # (M,3,n_rest)
            return torch.cat((dc, rest), dim=2)                  # (M,4,n_tot)
        else:
            # ─ legacy: tensors already (M, …); just return them
            return torch.cat((self._features_dc,
                            self._features_rest), dim=1)

    @property
    def get_asg_features(self):
        if getattr(self, "uniform_strand_color", False):
            return self._features_asg[self._strand_id]           # broadcast
        else:
            return self._features_asg

    @property
    def get_opacity(self):
        if getattr(self, "uniform_strand_color", False):
            return torch.sigmoid(self._opacity[self._strand_id]) # broadcast
        else:
            return torch.sigmoid(self._opacity)

    @property
    def get_axial_weight(self):           # s in [0,1]
        return torch.sigmoid(self._s)

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
        return opacity * coef[..., None]


    @property
    def get_radial_uv(self):
        """
        Return (u,v) offsets in metres – already clamped to max_radius (= m/2).
        """
        rho = torch.sigmoid(self._rho_hat)
        return self._max_radius * torch.cat([rho * torch.cos(self._phi),
                                            rho * torch.sin(self._phi)], dim=1)

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

    def perm2scene(self, points):
        """Convert centimetre vertices coming from PERM to metres."""
        return points / 100.0


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
        Interpolate centre positions and unit tangents along one strand.

        pts : (V, 3)  – strand vertices in metres
        s   : (K,)    – axial weights ∈ [0, 1]

        Returns
        -------
        centre  : (K, 3)
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
        centre = (1.0 - w) * P0 + w * P1

        T_raw  = P1 - P0                               # (K, 3) *or* (3,)
        tangent = torch.nn.functional.normalize(T_raw, dim=-1, eps=eps)

        return centre, tangent
    
    # ------------------------------------------------------------------
    def world_centers_and_tangents(self,
                                strand_vertices: torch.Tensor):
        """
        Compute the world-space centre *and* tangent of every Gaussian
        from canonical strand vertices – without touching self._xyz.

        Parameters
        ----------
        strand_vertices : (S,V,3) tensor in metres (canonical pose).

        Returns
        -------
        centers  : (M,3) tensor
        tangents : (M,3) tensor  (unit length)
        """
        if strand_vertices.dim() != 3:
            raise RuntimeError("strand_vertices must be (S,V,3)")
        if strand_vertices.shape[-1] != 3:                 # got (S,3,V)
            strand_vertices = strand_vertices.permute(0, 2, 1)

        S, V, _ = strand_vertices.shape
        M       = self._s.shape[0]

        sid   = self._strand_id                  # (M,)
        s_lin = torch.sigmoid(self._s)           # (M,)  axial weight [0,1]
        t     = s_lin * (V - 1)                  # (M,)
        idx0  = t.floor().long().clamp_max(V-2)  # left vertex
        idx1  = idx0 + 1                         # right vertex
        w     = (t - idx0.float()).unsqueeze(1)  # (M,1)

        P0 = strand_vertices[sid, idx0]          # (M,3)
        P1 = strand_vertices[sid, idx1]          # (M,3)
        Tcur = P1 - P0                           # (M,3)

        # ---- smooth tangent blend ---------------------------------------
        idx_prev = (idx0 - 1).clamp(min=0)
        idx_next = (idx1 + 1).clamp(max=V-1)
        Pprev = strand_vertices[sid, idx_prev]
        Pnext = strand_vertices[sid, idx_next]
        Tprev = P0 - Pprev
        Tnext = Pnext - P1
        Tprev[idx0 == 0]     = Tcur[idx0 == 0]
        Tnext[idx1 == V-1]   = Tcur[idx1 == V-1]
        blend0 = (1-w)*Tprev + w*Tcur
        blend1 = (1-w)*Tcur  + w*Tnext
        Traw   = (1-w)*blend0 + w*blend1
        tangent = torch.nn.functional.normalize(Traw, dim=1, eps=1e-9)

        # ---- radial offset ----------------------------------------------
        radii = self._strand_radius[sid].unsqueeze(1)         # (M,1)
        rho   = torch.sigmoid(self._rho_hat)                  # (M,1)
        phi   = self._phi                                     # (M,1)
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
        centers = centers_on_axis + u*n1 + v*n2               # (M,3)

        return centers, tangent

    def update_xyz_rot_scale(self, strand_vertices, *_):
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
        M       = self._s.shape[0]

        # ------------------------------------------------------------------ strand-length cache
        seg          = (strand_vertices[:, 1:] - strand_vertices[:, :-1]).norm(dim=-1)  # (S,V-1)
        self.strand_length = seg.sum(dim=1)                                            # (S,)

        # ------------------------------------------------------------------ axial interpolation
        sid  = self._strand_id                          # (M,)
        t    = self.get_axial_weight * (V - 1)          # (M,)
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
        rho   = self._activation(self._rho_hat)          # (M,1)
        phi   = self._phi                                # (M,1)

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

        if torch.isnan(xyz_new).any() or torch.isinf(xyz_new).any():
            print("WARNING: NaN/Inf in xyz – skipping update_xyz_rot_scale")
            return

        self._xyz      = xyz_new
        self._rotation = self._edge_dirs_to_quat(tangent)

        # ------------------------------------------------------------------ keep legacy scaling update (optional)
        base = self._scaling_base
        perp = base[:, [0]]
        para = base[:, [1]]                              # let caller ensure σ‖ ≥ σ⊥
        self._scaling = torch.cat([perp, para, perp], dim=1)

        # ------------------------------------------------------------------ final size check
        assert self._xyz.shape[0] == M, \
            f"Wrong M after update: got {self._xyz.shape[0]} vs {M}"
        
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
        thick_perp: float = 6.645693247264717e-05,  # default σ⊥
        par_ratio:  float = 20.0,                  # default σ‖ / σ⊥
        thresh_mul: float = 2.0                    # clamp when σ‖ ≥ 3× default
    ) -> int:
        """
        If σ‖ ≥ thick_perp * par_ratio * thresh_mul, divide that σ‖ by two.
        Returns the number of Gaussians affected.
        """
        # 1. Current longitudinal sigmas (σ‖ = exp(log σ‖))
        sigma_par = self._scaling_base[:, 1].exp()                # (M,)
        threshold = thick_perp * par_ratio * thresh_mul

        # 2. Offending Gaussians
        mask = sigma_par >= threshold
        n_fixed = int(mask.sum().item())
        if n_fixed == 0:
            return 0

        # 3. Halve σ‖  → subtract ln 2 from its log value
        self._scaling_base.data[mask, 1] -= math.log(2.0)
        self._scaling_base.data[mask, 0] -= math.log(2.0)

        # 4. Keep cached _scaling tensor in sync
        base  = self._scaling_base
        perp  = base[:, [0]]
        para  = base[:, [1]]
        self._scaling = torch.cat([perp, para, perp], dim=1)

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
    #  ----  CLONE (vanilla 3D-GS)  -----------------------------------
    # ================================================================
    @torch.no_grad()
    def densify_clone(self, grads: torch.Tensor,
                    grad_thresh: float,
                    scene_extent: float):
        """
        Duplicate Gaussians whose full-norm grads exceed grad_thresh and
        whose sigma⊥ is still *larger* than percent_dense × extent (same rule
        as vanilla GS).  Radial cage is copied unchanged.
        """
        g_norm = grads.norm(dim=-1)               # (M,)
        sel = (g_norm >= grad_thresh) & \
            (self.get_scaling.max(dim=1).values
                <= self.percent_dense * scene_extent)
        if sel.sum() == 0:
            return                                # nothing to clone

        # tensors to append ------------------------------------------------
        new_xyz      = self._xyz[sel]
        new_features_dc   = self._features_dc  [sel]
        new_features_rest = self._features_rest[sel]
        new_features_asg  = self._features_asg[sel]
        new_opacity  = self._opacity[sel]
        new_scaling  = self._scaling_base[sel]
        new_rotation = self._rotation[sel]

        new_s        = self._s[sel]
        new_rho_hat  = self._rho_hat[sel]
        new_phi      = self._phi[sel]
        new_sid      = self._strand_id[sel]

        # extend all learnable tensors + Adam state
        ext = self._cat_to_optimizer({
            "xyz":       new_xyz,
            "f_dc":      new_features_dc,
            "f_rest":    new_features_rest,
            "f_asg":     new_features_asg,
            "opacity":   new_opacity,
            "scaling":   new_scaling,
            "rotation":  new_rotation,
            "axial":     new_s,
            "rho":       new_rho_hat,
            "phi":       new_phi,
        })

        # re-bind class attributes to fresh Parameter handles
        self._xyz            = ext["xyz"]
        self._features_dc    = ext["f_dc"]
        self._features_rest  = ext["f_rest"]
        self._features_asg   = ext["f_asg"]
        self._opacity        = ext["opacity"]
        self._scaling_base   = ext["scaling"]
        self._rotation       = ext["rotation"]
        self._s              = ext["axial"]
        self._rho_hat        = ext["rho"]
        self._phi            = ext["phi"]

        # bookkeeping for strand look-ups
        self._strand_id = torch.cat((self._strand_id, new_sid), 0)
        self.num_gaussians = int(self._xyz.shape[0])

        # update prefix sums: +1 for each new entry in its strand
        add_per_strand = torch.bincount(new_sid,
                                        minlength=self.num_strands)
        self._gauss_offset[1:] += add_per_strand.cumsum(0)

        # reset accumulators
        self.xyz_gradient_accum = torch.zeros((self.num_gaussians, 1),
                                            device=self._xyz.device)
        self.denom = torch.zeros_like(self.xyz_gradient_accum)


    # ================================================================
    #  ----  AXIAL SPLIT (strand-aware)  -------------------------------
    # ================================================================
    @torch.no_grad()
    def densify_split(self, grads: torch.Tensor,
                    grad_thresh: float,
                    eps_axial: float = 0.04):
        """
        Split Gaussians with large **view-space gradient** into two children
        displaced ± eps_axial along the strand *axis* (s parameter).
        sigma∥ is halved; sigma⊥ unchanged.  Radial cage is copied.
        """
        g_val = grads.squeeze()
        sel   = g_val >= grad_thresh
        if sel.sum() == 0:
            return

        # child A (+eps)  and  child B ( –eps)
        s_parent   = self._s[sel]
        sid_parent = self._strand_id[sel]

        s_a = torch.clamp(s_parent + eps_axial, 0.0, 1.0)
        s_b = torch.clamp(s_parent - eps_axial, 0.0, 1.0)

        new_s        = torch.cat([s_a, s_b], 0)
        new_sid      = torch.cat([sid_parent, sid_parent], 0)

        # scales: halve sigma∥  (log-space subtract ln2)
        log_scale_parent = self._scaling_base[sel]
        log_scale_child  = log_scale_parent.clone()
        log_scale_child[:, 1] -= np.log(2.0)      # sigma∥ /2
        new_scaling = torch.cat([log_scale_child, log_scale_child], 0)

        # duplicate all other attrs
        dup = lambda x: torch.cat([x[sel], x[sel]], 0)
        ext = self._cat_to_optimizer({
            "xyz":       dup(self._xyz),          # dummy, will be recomputed next frame
            "f_dc":      dup(self._features_dc),
            "f_rest":    dup(self._features_rest),
            "f_asg":    dup(self._features_asg),
            "opacity":   dup(self._opacity),
            "scaling":   new_scaling,
            "rotation":  dup(self._rotation),
            "axial":     new_s,
            "rho":       dup(self._rho_hat),
            "phi":       dup(self._phi),
        })

        # re-bind
        self._xyz            = ext["xyz"]
        self._features_dc    = ext["f_dc"]
        self._features_rest  = ext["f_rest"]
        self._features_asg   = ext["f_asg"]
        self._opacity        = ext["opacity"]
        self._scaling_base   = ext["scaling"]
        self._rotation       = ext["rotation"]
        self._s              = ext["axial"]
        self._rho_hat        = ext["rho"]
        self._phi            = ext["phi"]

        # update strand id / prefix sums
        self._strand_id = torch.cat((self._strand_id, new_sid), 0)
        add_per_strand  = torch.bincount(new_sid,
                                        minlength=self.num_strands)
        self._gauss_offset[1:] += add_per_strand.cumsum(0)
        self.num_gaussians = int(self._xyz.shape[0])

        # reset accumulators
        self.xyz_gradient_accum = torch.zeros((self.num_gaussians, 1),
                                            device=self._xyz.device)
        self.denom = torch.zeros_like(self.xyz_gradient_accum)


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
    def prune_inactive(self, alpha_thresh: float = 0.01):
        """
        Remove Gaussians whose opacity < alpha_thresh.  All per-Gaussian
        buffers (optimizer tensors, _xyz, _rotation, _strand_id, max_radii2D)
        get the exact same mask.
        """
        # build the keep-mask once
        mask = (self.get_opacity.squeeze() >= alpha_thresh)
        if mask.all():
            return

        # 1) prune every optimizer-managed parameter
        new_params = self._prune_optimizer_rows(mask)

        # re-bind those back onto self (names match your training_setup groups)
        self._s             = new_params["axial"]
        self._rho_hat       = new_params["rho"]
        self._phi           = new_params["phi"]
        self._features_dc   = new_params["f_dc"]
        self._features_rest = new_params["f_rest"]
        self._features_asg  = new_params["f_asg"]
        self._opacity       = new_params["opacity"]
        self._scaling_base  = new_params["scaling"]
        # (θ and β live in different groups, they get returned but we ignore them)

        # 2) prune the frozen buffers with the *same* mask
        self._xyz        = self._xyz[mask]
        self._rotation   = self._rotation[mask]
        self._strand_id  = self._strand_id[mask]
        self.max_radii2D = self.max_radii2D[mask]

        # 3) rebuild your offset lookup
        counts = torch.bincount(self._strand_id, minlength=self.num_strands)
        self._gauss_offset[0] = 0
        self._gauss_offset[1:] = counts.cumsum(0)

        # 4) reset counters & gradient stats to the new M
        M = int(self._xyz.shape[0])
        self.num_gaussians       = M
        device = self._xyz.device
        self.xyz_gradient_accum  = torch.zeros((M,1), device=device)
        self.denom               = torch.zeros_like(self.xyz_gradient_accum)

    @torch.no_grad()
    def add_densification_stats(self, viewspace_pts: torch.Tensor,
                                visible: torch.BoolTensor,
                                occ_mask: torch.BoolTensor):
        """
        viewspace_pts.grad: (N_visible_candidates, 3) – gradients for Gaussians where occ_mask == True
        visible:            (N_visible_candidates,)   – mask within the occ_mask subset
        occ_mask:           (M,)                      – full-size mask of Gaussians rendered
        """
        # 1. Compute gradient norm in x-y
        grad_xy = torch.norm(viewspace_pts.grad[visible, :2], dim=-1, keepdim=True)  # (N_visible, 1)

        # 2. Get indices in full tensor that correspond to visible subset
        full_occ_indices = occ_mask.nonzero(as_tuple=False).squeeze(1)              # (N_visible_candidates,)
        update_indices = full_occ_indices[visible]                                  # (N_visible,)

        # 3. In-place updates to full tensors
        self.xyz_gradient_accum[update_indices] += grad_xy
        self.denom[update_indices]              += 1

    @torch.no_grad()
    def densify_and_prune(self, grad_thr, min_opac, extent, max_screen=None, radii=None):
        # 1) average gradient
        grads = self.xyz_gradient_accum / torch.clamp_min(self.denom, 1e-9)
        grads[torch.isnan(grads)] = 0.0
        g_norm = grads.squeeze()  # (M,)

        # 2) record original M₀ before cloning/splitting
        M0 = self._s.shape[0]
        scaling0 = self.get_scaling.max(dim=1).values
        clone_mask = (g_norm[:M0] >= grad_thr) & (scaling0 <= 6.645693247264717e-05)
        split_mask = (g_norm[:M0] >= grad_thr) & (scaling0 >= 6.645693247264717e-05 * 20 * 5.0)

        # 3) densify
        self._densify_clone_mask(grads, clone_mask, extent)
        self._densify_split_mask(split_mask, grad_thr, original_M=M0)

        # 4) optional radius update _only_ on the ORIGINAL M₀
        if radii is not None:
            if radii.shape[0] != M0:
                raise RuntimeError(f"radii must have length {M0}, got {radii.shape[0]}")
            self.max_radii2D[:M0] = torch.maximum(self.max_radii2D[:M0], radii)

        # 5) single-pass prune — this will slice _every_ per-Gaussian buffer
        self.prune_inactive(min_opac)

        # 6) prune gaussians that are too large
        
        M1 = self._s.shape[0]
        self.num_gaussians = M1

        # 7) Sanity check
        assert (
            self._strand_id.shape[0] == M1
            == self._xyz.shape[0]
            == self.max_radii2D.shape[0]
        ), f"Post‐prune mismatch: s={M1}, id={self._strand_id.shape[0]}, xyz={self._xyz.shape[0]}, radii={self.max_radii2D.shape[0]}"

        return torch.count_nonzero(clone_mask), torch.count_nonzero(split_mask)

    @torch.no_grad()
    def _densify_clone_mask(self, grads, sel_mask, scene_extent):
        """
        Clone only those Gaussians where sel_mask==True.
        Mirrors your old densify_clone, but uses the externally-computed sel_mask.
        """
        sel = sel_mask
        if sel.sum() == 0:
            return

        # gather existing rows to duplicate
        new_xyz           = self._xyz[sel]
        new_features_dc   = self._features_dc[sel]
        new_features_rest = self._features_rest[sel]
        new_features_asg = self._features_asg[sel]
        new_opacity       = self._opacity[sel]
        new_scaling_base  = self._scaling_base[sel]
        new_rotation      = self._rotation[sel]
        new_s             = self._s[sel]
        new_rho_hat       = self._rho_hat[sel]
        new_phi           = self._phi[sel]
        new_strand_id     = self._strand_id[sel]

        # extend all learnables + Adam state
        ext = self._cat_to_optimizer({
            "xyz":      new_xyz,
            "f_dc":     new_features_dc,
            "f_rest":   new_features_rest,
            "f_asg":    new_features_asg,
            "opacity":  new_opacity,
            "scaling":  new_scaling_base,
            "rotation": new_rotation,
            "axial":    new_s,
            "rho":      new_rho_hat,
            "phi":      new_phi,
        })

        # re-bind to class
        self._xyz           = ext["xyz"]
        self._features_dc   = ext["f_dc"]
        self._features_rest = ext["f_rest"]
        self._features_asg = ext["f_asg"]
        self._opacity       = ext["opacity"]
        self._scaling_base  = ext["scaling"]
        self._rotation      = ext["rotation"]
        self._s             = ext["axial"]
        self._rho_hat       = ext["rho"]
        self._phi           = ext["phi"]

        # update strand IDs & prefix sums
        self._strand_id = torch.cat([self._strand_id, new_strand_id], dim=0)
        add_per_strand  = torch.bincount(new_strand_id, minlength=self.num_strands)
        self._gauss_offset[1:] += add_per_strand.cumsum(0)
        self.num_gaussians = int(self._xyz.shape[0])

        # reset accumulators
        self.xyz_gradient_accum = torch.zeros((self.num_gaussians, 1), device=self._xyz.device)
        self.denom              = torch.zeros_like(self.xyz_gradient_accum)
    
    @torch.no_grad()
    def _densify_split_mask(self, sel_mask, grad_thresh, original_M):
        """
        Split only the ORIGINAL_M Gaussians indicated by sel_mask.
        Any clones added in between are untouched.
        """
        sel = sel_mask
        if sel.sum() == 0:
            return

        # work on first original_M rows
        s_parent    = self._s[:original_M][sel]
        strand_ids  = self._strand_id[:original_M][sel]
        log_base    = self._scaling_base[:original_M][sel]
        strand_L    = self.strand_length[strand_ids]

        # half (1.6 based on og gs) the parallel sigma in log-space
        log_child     = log_base.clone()
        log_child -= np.log(2.0)
        new_scalings  = torch.cat([log_child, log_child], dim=0)

        # two child positions along s
        sigma_par_child     = log_child[:, 1].exp() / 2             # (P,)
        delta_s             = (sigma_par_child / strand_L).clamp_max(1.0)  # (P,)
        s_lin_parent = torch.sigmoid(s_parent)                  # (P,)
        s_lin_a = (s_lin_parent + delta_s).clamp(0.0, 1.0)
        s_lin_b = (s_lin_parent - delta_s).clamp(0.0, 1.0)

        # back to logits
        s_a = self.inverse_rho_activation(s_lin_a)
        s_b = self.inverse_rho_activation(s_lin_b)
        new_s = torch.cat([s_a, s_b], dim=0)
        new_ids    = torch.cat([strand_ids, strand_ids], dim=0)

        # duplicate all other attrs from the original slice
        def dup(x):
            chunk = x[:original_M][sel]
            return torch.cat([chunk, chunk], dim=0)

        ext = self._cat_to_optimizer({
            "xyz":      dup(self._xyz),
            "f_dc":     dup(self._features_dc),
            "f_rest":   dup(self._features_rest),
            "f_asg":    dup(self._features_asg),
            "opacity":  dup(self._opacity),
            "scaling":  new_scalings,
            "rotation": dup(self._rotation),
            "axial":    new_s,
            "rho":      dup(self._rho_hat),
            "phi":      dup(self._phi),
        })

        # re-bind attributes
        self._xyz           = ext["xyz"]
        self._features_dc   = ext["f_dc"]
        self._features_rest = ext["f_rest"]
        self._features_asg = ext["f_asg"]
        self._opacity       = ext["opacity"]
        self._scaling_base  = ext["scaling"]
        self._rotation      = ext["rotation"]
        self._s             = ext["axial"]
        self._rho_hat       = ext["rho"]
        self._phi           = ext["phi"]

        # append new strand IDs & update offsets
        self._strand_id = torch.cat([self._strand_id, new_ids], dim=0)
        add_per_strand  = torch.bincount(new_ids, minlength=self.num_strands)
        self._gauss_offset[1:] += add_per_strand.cumsum(0)
        self.num_gaussians = int(self._xyz.shape[0])

        # reset accumulators
        self.xyz_gradient_accum = torch.zeros((self.num_gaussians, 1), device=self._xyz.device)
        self.denom              = torch.zeros_like(self.xyz_gradient_accum)