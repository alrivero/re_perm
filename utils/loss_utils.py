#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp, pi
from typing import Optional

from utils.general_utils import project_to_screen, quaternion_to_rotation_matrix, convert_normal_to_camera_space


def huber_loss(network_output, gt, mask, delta, reduction='mean'):
    """
    delta: the Huber‐transition point
    mask: a weighting mask (e.g., hair alpha mask), same shape as input (H, W, 1) or (H, W)
    reduction: 'none' | 'sum' | 'mean'
    """
    diff = network_output - gt
    abs_diff = diff.abs()
    huber = torch.where(
        abs_diff < delta,
        0.5 * diff**2,
        delta * (abs_diff - 0.5 * delta)
    )

    # Ensure mask has the same shape as the loss
    if mask.ndim == 2:
        mask = mask.unsqueeze(-1)  # (H, W, 1)
    if huber.shape != mask.shape:
        mask = mask.expand_as(huber)

    # Apply mask
    weighted_loss = huber * mask

    if reduction == 'none':
        return weighted_loss
    elif reduction == 'sum':
        return weighted_loss.sum()
    elif reduction == 'mean':
        # Avoid dividing by total number of pixels if mask is sparse
        return weighted_loss.sum() / (mask.sum() + 1e-8)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

# def huber_loss(network_output, gt, alpha):
#     diff = torch.abs(network_output - gt)
#     mask = (diff < alpha).float()
#     loss = 0.5*diff**2*mask + alpha*(diff-0.5*alpha)*(1.-mask)
#     return loss.mean()

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def _project_gaussians_to_uv(gaussians, viewpoint_cam, H, W):
    """
    Returns integer pixel coordinates (u,v) and a boolean mask saying which
    gaussians lie inside the viewport.
    """
    xyz = gaussians.get_xyz                         # (N,3)
    uv  = project_to_screen(
            xyz,
            viewpoint_cam.projection_matrix[:3, :3].unsqueeze(0),
            viewpoint_cam.w2c.unsqueeze(0)
          )[0].long()                               # (N,2) long

    u, v = uv[:, 0], uv[:, 1]
    in_view = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u[in_view], v[in_view], in_view          # filtered u,v and mask


def orientation_loss_kernel(
    viewpoint_cam,
    strand_pts : torch.Tensor,
    hair_mask  : torch.Tensor,
    orient_map : torch.Tensor,
    conf_thresh: float = 0.05,
    radius     : int   = 3,
    sigma      : Optional[float] = None,   # ← change here
) -> torch.Tensor:
    """
    Each strand edge projects its mid‑point; a (2r+1)² window from the
    orientation map is compared to the edge’s projected 2‑D direction
    with a Gaussian weighting (σ defaults to 0.5·radius).
    """

    if sigma is None: sigma = max(1.0, 0.5 * radius)
    device = strand_pts.device

    # ---------- orientation map to (H,W,2) ----------------------------
    om = orient_map
    if om.dim() == 3 and om.shape[0] <= 3:
        om = om.permute(1, 2, 0).contiguous()          # (H,W,C)
    if om.shape[2] > 2:
        om = om[:, :, :2]
    ang_deg = om[..., 0] * 255.0                       # restore degrees
    conf    = om[..., 1]
    H, W    = ang_deg.shape

    # ---------- edge mid‑points & directions (world) ------------------
    mids = 0.5 * (strand_pts[:, 1:] + strand_pts[:, :-1]).reshape(-1, 3)    # (M,3)
    dirs = (strand_pts[:, 1:] - strand_pts[:, :-1]).reshape(-1, 3)          # (M,3)

    # ---------- project mid‑points to pixels --------------------------
    px = project_to_screen(
            mids,
            viewpoint_cam.projection_matrix[:3, :3].unsqueeze(0),
            viewpoint_cam.w2c.unsqueeze(0)
        )[0]                                            # (M,2) float
    u = px[:, 0].round().long().clamp(0, W - 1)
    v = px[:, 1].round().long().clamp(0, H - 1)

    # keep only pixels in front of the camera & inside mask
    in_mask = hair_mask[0, v, u] > 0.0
    if in_mask.sum() == 0:
        return torch.tensor(0.0, device=device)
    u, v, dirs = u[in_mask], v[in_mask], dirs[in_mask]  # (M',)

    M = u.numel()

    # ---------- camera‑space 2‑D directions ---------------------------
    dirs_cam = convert_normal_to_camera_space(
                  dirs,
                  viewpoint_cam.w2c[:3, :3],
                  viewpoint_cam.projection_matrix[:3, :3])
    dirs_2d  = F.normalize(dirs_cam[:, :2], dim=-1)     # (M,2)

    # ---------- Gaussian kernel ---------------------------------------
    K  = 2 * radius + 1
    offs = torch.arange(-radius, radius + 1, device=device)
    dy, dx = torch.meshgrid(offs, offs, indexing='ij')
    ker = torch.exp(-(dx**2 + dy**2) / (2 * sigma * sigma)).flatten()  # (K²,)

    # ---------- build window indices (vectorised) ---------------------
    uu = u.view(M, 1) + dx.flatten().view(1, -1)        # (M,K²)
    vv = v.view(M, 1) + dy.flatten().view(1, -1)
    uu = uu.clamp(0, W - 1)
    vv = vv.clamp(0, H - 1)

    lin = vv * W + uu                                   # (M,K²) linear idx
    ang_win  = ang_deg.view(-1)[lin]                    # (M,K²)
    conf_win = conf.view(-1)[lin]                       # (M,K²)
    mask_win = hair_mask.view(-1)[lin] > 0              # (M,K²)

    valid = (conf_win > conf_thresh) & mask_win         # (M,K²)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)

    theta = (ang_win * pi / 180.0)                      # radians
    gt_vec = torch.stack((torch.sin(theta), torch.cos(theta)), dim=-1)  # (M,K²,2)

    pred   = dirs_2d.unsqueeze(1)                       # (M,1,2)
    cos    = (pred * gt_vec).sum(-1).clamp(-1., 1.)     # (M,K²)
    invcos = 1.0 - cos

    w = ker.view(1, -1) * conf_win * valid.float()      # (M,K²)
    loss_per_edge = (w * invcos).sum(1) / (w.sum(1) + 1e-8)

    finite = torch.isfinite(loss_per_edge)
    return loss_per_edge[finite].mean() if finite.any() else torch.tensor(0.0, device=device)

def orientation_loss(
        viewpoint_cam,
        gaussians,
        hair_mask,          # (1,H,W) float {0,1}
        orient_map,         # either (H,W,2) or (C,H,W), normalized [0,1]
        conf_thresh=0.05
    ):
    """
    Angular loss between rendered Gaussian orientations and a ground-truth
    orientation map loaded via PILtoTensor (so both channels in [0,1]).
    
    Channel-0 is angle (8-bit stored → denormalized to [0,255]°)
    Channel-1 is confidence [0,1].
    """

    device = gaussians.get_xyz.device

    # 0) reshape to (H, W, C)
    om = orient_map
    if om.dim() == 3:
        # (C,H,W) → (H,W,C)
        om = om.permute(1, 2, 0).contiguous()
    # drop any extra channels (e.g. blue)
    if om.shape[2] > 2:
        om = om[:, :, :2]

    # 0b) denormalize angle channel: [0,1]→[0,255] degrees
    angle_norm = om[..., 0]
    conf_map   = om[..., 1]
    angle_deg  = angle_norm * 255.0
    om = torch.stack([angle_deg, conf_map], dim=-1)  # back to (H,W,2)

    H, W = om.shape[:2]

    # 1) project Gaussians into pixel coords
    u, v, in_view = _project_gaussians_to_uv(gaussians, viewpoint_cam, H, W)
    if u.numel() == 0:
        return torch.tensor(0.0, device=device)

    idx_g = in_view.nonzero(as_tuple=False).squeeze(1)

    # 2) keep only those that fall on the hair mask
    on_hair = hair_mask[0, v, u] > 0.0
    if on_hair.sum() == 0:
        return torch.tensor(0.0, device=device)

    u, v    = u[on_hair], v[on_hair]
    idx_g   = idx_g[on_hair]

    # 3) discard low-confidence pixels
    gt_ang   = om[v, u, 0]
    gt_conf  = om[v, u, 1]
    keep     = gt_conf > conf_thresh
    if keep.sum() == 0:
        return torch.tensor(0.0, device=device)

    u, v     = u[keep], v[keep]
    idx_g    = idx_g[keep]
    gt_rad   = gt_ang[keep] * pi / 180.0           # convert to radians

    # 4) ground-truth unit vectors
    gt_vec2d = torch.stack((torch.sin(gt_rad),
                            torch.cos(gt_rad)), dim=-1)     # (M,2)

    # 5) predicted directions from Gaussians
    R_local    = quaternion_to_rotation_matrix(gaussians.get_rotation)  # (N,3,3)
    y_world    = R_local[:, :, 1]                                       # (N,3)
    dirs_world = y_world[idx_g]                                         # (M,3)

    dirs_cam   = convert_normal_to_camera_space(
                     dirs_world,
                     viewpoint_cam.w2c[:3, :3],
                     viewpoint_cam.projection_matrix[:3, :3])
    dirs_2d    = F.normalize(dirs_cam[:, :2], dim=-1)                   # (M,2)

    # 6) cosine-based loss (1 − cos Δθ)
    inv_dot   = 1.0 - (dirs_2d * gt_vec2d).sum(dim=-1).clamp(-1., 1.)    # (M,)

    # optional scale weighting
    scales      = gaussians.get_scaling                                # (N,3)
    scale_ratio = (scales[idx_g, 0] / scales[idx_g, 1]).clamp_min(1e-6)
    weight      = torch.exp(-scale_ratio)                              # (M,)

    loss = (weight * inv_dot).mean()
    return loss.to(device)

def strand_length_loss(strand_points,
                       L_max   = 0.10,   # metres (== 10 cm in scene space)
                       delta   = 0.01):  # 1 cm smooth hinge
    """
    strand_points : (S,V+1,3) tensor – vertices of every strand
    Returns a scalar.
    """
    seg_len   = (strand_points[:, 1:] - strand_points[:, :-1]).norm(dim=-1)  # (S,V)
    total_len = seg_len.sum(dim=-1)                                          # (S,)

    excess    = F.relu(total_len - L_max)                                    # (S,)
    # Huber hinge on the excess
    loss = torch.where(excess < delta,
                       0.5 * excess**2 / delta,
                       excess - 0.5 * delta)
    return loss.mean()

def neighbour_orientation_loss(strand_points,
                               k             = 8,      # how many neighbours per strand
                               eps           = 1e-6):  # numerical safety
    """
    strand_points : (S,V,3)  tensor  (S = num_strands, V = STRAND_VERTEX_COUNT)
                     *must* contain the root at index‑0 and the first off‑root
                     vertex at index‑1 so we can define an orientation vector
                     along the strand.

    Returns a scalar loss in [0,2].
    """
    S = strand_points.size(0)

    # ------------------------------------------------------------------ #
    # 1 ) orientation vector of every strand (root → vertex‑1)           #
    # ------------------------------------------------------------------ #
    orient = strand_points[:, 1] - strand_points[:, 0]         # (S,3)
    orient = F.normalize(orient, dim=-1, eps=eps)              # (S,3)

    # ------------------------------------------------------------------ #
    # 2 ) find spatial neighbours using root positions                  #
    # ------------------------------------------------------------------ #
    roots = strand_points[:, 0]                                # (S,3)
    # pairwise squared distance matrix  (S,S)
    d2 = torch.cdist(roots, roots, p=2)**2                     # (S,S)

    # k+1 because distance to itself == 0
    _, idx_knn = torch.topk(d2, k=k+1, largest=False)          # (S,k+1)
    idx_knn = idx_knn[:, 1:]                                   # drop self → (S,k)

    # ------------------------------------------------------------------ #
    # 3 ) cosine similarity with each neighbour                         #
    # ------------------------------------------------------------------ #
    v_i   = orient.unsqueeze(1).expand(-1, k, -1)              # (S,k,3)
    v_j   = orient[idx_knn]                                    # (S,k,3)

    cos_ij = (v_i * v_j).sum(dim=-1).clamp(-1.0, 1.0)          # (S,k)

    # ------------------------------------------------------------------ #
    # 4 ) loss  = 1 – cosθ   (0 when perfectly aligned, ↑ as they diverge)
    # ------------------------------------------------------------------ #
    loss = (1.0 - cos_ij).mean()                               # scalar
    return loss

def outside_opacity_loss(viewpoint_cam, gaussians, hair_mask):
    """
    Encourage Gaussians to either stay inside the hair mask or pay a cost
    if they try to dodge coverage by zeroing their opacity.

    Args:
        viewpoint_cam : your camera object
        gaussians     : GaussianPerm instance
        hair_mask     : (1,H,W) float mask where 1=hair, 0=background

    Returns:
        scalar loss on the same device as the Gaussians
    """
    device = gaussians.get_xyz.device
    H, W = hair_mask.shape[1:]

    # 1) project every Gaussian’s centre into pixel coords
    u, v, in_view = _project_gaussians_to_uv(gaussians, viewpoint_cam, H, W)
    if u.numel() == 0:
        return torch.tensor(0.0, device=device)

    # indices of Gaussians that are in view
    idxs = in_view.nonzero(as_tuple=False).squeeze(1)

    # 2) find which of those land outside the hair mask
    outside = (hair_mask[0, v, u] < 0.5)
    if outside.sum() == 0:
        return torch.tensor(0.0, device=device)

    # 3) fetch their opacities
    opacity = gaussians.get_opacity.squeeze(-1)   # (N,)
    opac_out = opacity[idxs[outside]]             # (M,)

    # 4) penalize “shutting off” → high when opacity→0, zero when opacity=1
    loss = (1.0 - opac_out).mean()
    return loss

def orientation_match_strands_loss(
    strands: torch.Tensor,       # (S, N, 3)
    gaussians,                       # your GaussianPerm instance
) -> torch.Tensor:
    """
    Enforce each Gaussian’s rotation to align its local +Y with the strand segment.
    """
    # --- build edges & unit‐dirs ---
    edges = strands[:, 1:, :] - strands[:, :-1, :]        # (S, N-1, 3)
    M = edges.numel() // 3
    edge_dirs = edges.reshape(M, 3)
    edge_lens = edge_dirs.norm(dim=1, keepdim=True).clamp(min=1e-6)
    dirs = edge_dirs / edge_lens                            # (M,3)

    # --- get live quaternions and normalize ---
    quats = gaussians.get_rotation  # (M,4)
    w, x, y, z = quats.unbind(dim=1)

    # rotate +Y -> v_pred
    v_pred = torch.stack([
        2*(x*y + w*z),
        w*w - x*x - z*z + y*y,
        2*(y*z - w*x),
    ], dim=1)  # (M,3)

    # cosine similarity, clamp negative to zero
    cosines = (v_pred * dirs).sum(dim=1).clamp(min=0.0)
    return (1 - cosines).mean()


def oblong_shape_loss_from_strands_loss(
    strands: torch.Tensor,       # (S, N, 3)
    gaussians,                       # your GaussianPerm instance
) -> torch.Tensor:
    """
    Push x/z scales to be much smaller than y scales.
    """
    # --- build edges & lengths for scale targets (unused) ---
    edges = strands[:, 1:, :] - strands[:, :-1, :]
    M = edges.numel() // 3

    # --- get live scales ---
    scales = gaussians.scaling_activation(gaussians._scaling_base)  # (M,2)
    xz = scales[:, 0]
    y  = scales[:, 1].clamp(min=1e-6)

    # penalize (xz / y)^2
    return ( (xz / y) ** 2 ).mean()


def length_consistency_loss_from_strands_loss(
    strands: torch.Tensor,       # (S, N, 3)
    gaussians,                       # your GaussianPerm instance
) -> torch.Tensor:
    """
    Make each Gaussian’s y‐scale equal the true segment length.
    """
    # --- build edges & true lengths ---
    edges = strands[:, 1:, :] - strands[:, :-1, :]
    M = edges.numel() // 3
    edge_dirs = edges.reshape(M, 3)
    edge_lens = edge_dirs.norm(dim=1, keepdim=True)        # (M,1)

    # --- get live y‐scale ---
    scales = gaussians.scaling_activation(gaussians._scaling_base)  # (M,2)
    scale_y = scales[:, 1].unsqueeze(1)                     # (M,1)

    return ((scale_y - edge_lens) ** 2).mean()

def bending_loss(strand_pts: torch.Tensor) -> torch.Tensor:
    """
    Second‐derivative smoothness along each strand.
    strand_pts: (S, N, 3) array of the strand control points.
    Penalizes ||p[i+2] − 2p[i+1] + p[i]||^2.
    """
    # take second finite difference along the N dimension
    dd = strand_pts[:, 2:, :] - 2.0 * strand_pts[:, 1:-1, :] + strand_pts[:, :-2, :]
    # dd is (S, N-2, 3)
    return (dd.norm(dim=2) ** 2).mean()

def neighbor_scale_smoothness_loss(strand_pts: torch.Tensor, perm) -> torch.Tensor:
    """
    Encourage the Gaussians’ thickness to vary slowly along each strand.
    """
    # rebuild edges‐count M = S*(N−1), same as in init
    S, N, _ = strand_pts.shape
    M = S * (N - 1)
    # live scales: (M,2)  → reshape to (S, N-1, 2)
    scales = perm.scaling_activation(perm._scaling_base).view(S, N - 1, 2)
    # neighboring differences along the strand
    ds = scales[:, 1:, :] - scales[:, :-1, :]      # (S, N-2, 2)
    return (ds.pow(2).mean())

def aligned_depth_loss(
    D_pred      : torch.Tensor,   # (H,W) renderer depth in world units
    D_gt        : torch.Tensor,   # (H,W) HairStep depth   (0‒1 float or 8‑bit) 
    mask        : torch.Tensor,   # (H,W) bool / 0‑1 hair alpha
    mode:str    = "si_rmse",      # "si_rmse"  or  "huber"
    huber_beta : float = 0.01,    # only used when mode=="huber"
    eps        : float = 1e-3     # numerical stabiliser for log
):
    """
    Aligns D_pred to D_gt with a per‑frame scale/shift and returns a scale‑invariant loss.

    Returns
    -------
    loss : torch.Tensor  (scalar)
    D_aligned : torch.Tensor  (H,W)  -- for visualisation/debug
    """
    # -------- 1. fit a,b s.t. a*D_pred + b ≈ D_gt over hair pixels --------
    # import pdb; pdb.set_trace()
    P = D_pred[mask].view(-1)
    G = D_gt  [mask].view(-1)


    denom = (P*P).sum().clamp(min=1e-6)
    a = (P*G).sum() / denom
    b = G.mean() - a * P.mean()

    D_align = a * D_pred + b

    # -------- 2. loss options --------------------------------------------
    if mode.lower() == "huber":
        loss = F.smooth_l1_loss(D_align[mask], G, beta=huber_beta)
    elif mode.lower() == "si_rmse":
        # scale‑invariant RMSE from Eigen et al. ’14 (log space)
        log_d = torch.log(D_align[mask] + eps) - torch.log(G + eps)
        loss = (log_d.pow(2).mean() - log_d.mean() ** 2).sqrt()
    else:
        raise ValueError("mode must be 'si_rmse' or 'huber'")

    return loss, D_align