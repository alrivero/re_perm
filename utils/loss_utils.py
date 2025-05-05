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

from utils.general_utils import project_to_screen, quaternion_to_rotation_matrix, convert_normal_to_camera_space


def huber_loss(network_output, gt, delta, reduction='mean'):
    """
    delta: the Huber‐transition point
    reduction: 'none' | 'sum' | 'mean'
    """
    diff = network_output - gt
    abs_diff = diff.abs()
    mask = (abs_diff < delta)

    # elementwise Huber:
    loss = torch.where(
        mask,
        0.5 * diff**2,
        delta * (abs_diff - 0.5 * delta)
    )

    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
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


def orientation_loss(
        viewpoint_cam,
        gaussians,
        hair_mask,          # (1,H,W)  float {0,1}
        orient_map,         # (H,W,2)  channel‑0: angle[deg], channel‑1: confidence[0‑1]
        conf_thresh=0.05
    ):
    """
    Angular loss between projected gaussian orientations and a ground‑truth
    orientation map.  Pixels with confidence ≤ conf_thresh are ignored.

    Returns a **scalar** tensor (device = gaussians.device).
    """

    device = gaussians.get_xyz.device
    H, W   = orient_map.shape[:2]

    # ------------------------------------------------------------------ #
    # 1 ) project gaussians to the image plane                            #
    # ------------------------------------------------------------------ #
    u, v, mask_in_view = _project_gaussians_to_uv(gaussians, viewpoint_cam, H, W)
    if u.numel() == 0:          # nothing visible
        return torch.tensor(0.0, device=device)

    idx_g = mask_in_view.nonzero(as_tuple=False).squeeze(1)      # indices of visible gaussians

    # ------------------------------------------------------------------ #
    # 2 ) keep only those landing on *hair‑mask* pixels                  #
    # ------------------------------------------------------------------ #
    on_hair = hair_mask[0, v, u] > 0.0
    if on_hair.sum() == 0:
        return torch.tensor(0.0, device=device)

    u, v    = u[on_hair], v[on_hair]
    idx_g   = idx_g[on_hair]

    # ------------------------------------------------------------------ #
    # 3 ) discard low‑confidence orientation samples                     #
    # ------------------------------------------------------------------ #
    gt_angle_deg  = orient_map[v, u, 0]
    gt_confidence = orient_map[v, u, 1]
    keep          = gt_confidence > conf_thresh
    if keep.sum() == 0:
        return torch.tensor(0.0, device=device)

    u, v           = u[keep], v[keep]
    idx_g          = idx_g[keep]
    gt_angle_rad   = gt_angle_deg[keep] * pi / 180.0              # (M,)

    # ------------------------------------------------------------------ #
    # 4 ) ground‑truth 2‑D unit vectors (sinθ, cosθ)                     #
    # ------------------------------------------------------------------ #
    gt_vec2d = torch.stack((torch.sin(gt_angle_rad),
                            torch.cos(gt_angle_rad)), dim=-1)     # (M,2)

    # ------------------------------------------------------------------ #
    # 5 ) predicted orientation on the image plane                       #
    # ------------------------------------------------------------------ #
    # local +Y axis in world space
    R_local      = quaternion_to_rotation_matrix(gaussians.get_rotation)  # (N,3,3)
    y_axis_world = R_local[:, :, 1]                                       # (N,3)
    dirs_world   = y_axis_world[idx_g]                                    # (M,3)

    # world → camera → normalise & drop z
    dirs_cam     = convert_normal_to_camera_space(
                        dirs_world,
                        viewpoint_cam.w2c[:3, :3],
                        viewpoint_cam.projection_matrix[:3, :3])
    dirs_2d      = F.normalize(dirs_cam[:, :2], dim=-1)                   # (M,2)

    # ------------------------------------------------------------------ #
    # 6 ) angle‑based loss (1 − cos Δθ), optionally weighted              #
    # ------------------------------------------------------------------ #
    inv_dot = 1.0 - (dirs_2d * gt_vec2d).sum(dim=-1).clamp(-1., 1.)       # (M,)

    # scale‑ratio weighting (optional, comment out if not needed)
    scales        = gaussians.get_scaling                                # (N,3)
    scale_ratio   = (scales[idx_g, 0] / scales[idx_g, 1]).clamp_min(1e-6)
    weight        = torch.exp(-scale_ratio)                               # (M,)
    loss          = (weight * inv_dot).mean()                             # scalar

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