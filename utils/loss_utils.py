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
import lpips
import math
from torch.special import digamma, gammaln
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp, pi
from typing import Optional, Tuple
from pytorch3d.ops import knn_points 
from pytorch3d.structures import Meshes

from utils.general_utils import project_to_screen, quaternion_to_rotation_matrix, convert_normal_to_camera_space
from kornia.filters import sobel, gaussian_blur2d
from kornia.losses import ssim_loss as ms_ssim, ssim_loss

# ------------------------------------------------------------------ helpers
def _huber(x: torch.Tensor,
           y: torch.Tensor,
           delta: float = 0.1,
           reduction: str = "mean") -> torch.Tensor:
    """Classic Huber / smooth-L1 loss with custom delta."""
    diff  = x - y
    abs_d = diff.abs()
    loss  = torch.where(abs_d <= delta,
                        0.5 * diff.pow(2),
                        delta * (abs_d - 0.5 * delta))
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    return loss.mean()                 # default == "mean"

# ------------------------------------------------------------------ main
def alpha_blended_loss(pred_alpha: torch.Tensor,
                       tgt_alpha : torch.Tensor,
                       *,
                       delta: float = 0.00,
                       w_l1 : float = 1.0,
                       w_ssim: float = 0.0,
                       w_grad: float = 0.00) -> torch.Tensor:
    """
    Composite α-matte loss:

        L = w_l1   · Huber(pred, tgt)
          + w_ssim · MS-SSIM(pred, tgt)
          + w_grad · L1(∇pred, ∇tgt)

    Shapes
    ------
    pred_alpha, tgt_alpha : (B, 1, H, W) in [0, 1]

    Returns
    -------
    scalar loss (torch.Tensor, shape = ())
    """

    # --- make sure inputs are single-channel (B,1,H,W) float32 ------------
    if pred_alpha.ndim == 3:   # (B,H,W)
        pred_alpha = pred_alpha.unsqueeze(1)
    if tgt_alpha.ndim == 3:
        tgt_alpha = tgt_alpha.unsqueeze(1)
    pred_alpha = pred_alpha.float()
    tgt_alpha  = tgt_alpha.float()

    # --- component losses -------------------------------------------------
    loss_l1 = F.l1_loss(pred_alpha, tgt_alpha)

    # kornia ms-SSIM returns the dissimilarity (1 − MS-SSIM)
    loss_ssim  = ssim_loss(pred_alpha, tgt_alpha, window_size=11, reduction="mean")

    # gradient L1 via Sobel (produces (B,2,H,W); we take magnitude)
    grad_pred = sobel(pred_alpha)      # (B, 2, H, W)
    grad_tgt  = sobel(tgt_alpha)
    loss_grad = F.l1_loss(grad_pred, grad_tgt)

    # --- weighted sum -----------------------------------------------------
    loss =  w_l1 * loss_l1 \
          + w_ssim * loss_ssim \
          + w_grad * loss_grad
    return loss

import math
import torch
import torch.nn.functional as F
from torch.special import gammaln

import math
import torch
import torch.nn.functional as F
from torch.special import gammaln

class MaskShapeKLWithExtras:
    r"""
    1) KL(mask_pred‖template) using the old “spike+Beta” recipe (non-zero only).
    2) Soft‐Dice loss + κ schedule (same as before).
    3) One-sided exponential penalty for dropping below target non-zero fraction.
    """

    def __init__(
        self,
        # --- KL template params ---
        pi_star: float,       # desired spike mass
        alpha_star: float,    # Beta α
        beta_star: float,     # Beta β
        spike_thresh: float = 254/255,
        # --- KL annealing (β‐anneal) ---
        beta_start: float = 1e-3,
        beta_final: float = 1e-4,
        t_start:    int   = 8000,
        t_end:      int   = 50_000,

        # --- Dice & κ schedule ---
        dice_weight:  float = 1.0,
        warmup_iter:  int   = 15_000,
        kappa_lambda: float = 2e-4,

        # --- non-zero fraction penalty ---
        target_nz_frac: float = 0.25,
        lambda_nz:      float = 1.0,
        nz_exp_factor:  float = 5.0,
        nz_thresh:      float = 1e-3,
        nz_soft_width:  float = 0.01,

        eps:          float = 1e-6
    ):
        # KL template
        self.pi_star    = pi_star
        self.alpha_star = alpha_star
        self.beta_star  = beta_star
        self.spike_thr  = spike_thresh
        self.eps        = eps
        # precompute log B(α*,β*)
        self.log_B_star = (
            gammaln(torch.tensor(alpha_star))
          + gammaln(torch.tensor(beta_star))
          - gammaln(torch.tensor(alpha_star + beta_star))
        ).item()

        # KL anneal
        self.beta_start = beta_start
        self.beta_final = beta_final
        self.t_start    = t_start
        self.t_end      = max(t_end, t_start + 1)

        # Dice + κ
        self.dice_weight  = dice_weight
        self.warmup_iter  = warmup_iter
        self.kappa_lambda = kappa_lambda

        # non-zero fraction penalty
        self.target_nz_frac = target_nz_frac
        self.lambda_nz      = lambda_nz
        self.nz_exp_factor  = nz_exp_factor
        self.nz_thresh      = nz_thresh
        self.nz_soft_width  = nz_soft_width

    def _beta_pdf_log(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.alpha_star, self.beta_star
        return (
            (a - 1.0) * torch.log(torch.clamp(x, self.eps, 1.0))
          + (b - 1.0) * torch.log(torch.clamp(1.0 - x, self.eps, 1.0))
          - self.log_B_star
        )

    def _kl_anneal(self, step: int) -> float:
        if step < self.t_start:  return self.beta_start
        if step >= self.t_end:   return self.beta_final
        t = (step - self.t_start) / (self.t_end - self.t_start)
        return (1 - t) * self.beta_start + t * self.beta_final

    @staticmethod
    def _soft_dice(pred: torch.Tensor, gt: torch.Tensor, eps: float) -> torch.Tensor:
        p = pred.reshape(-1)
        g = gt.reshape(-1)
        inter = (p * g).sum()
        return 2 * inter / (p.sum() + g.sum() + eps)

    def _kappa_from_dice(self, dice_val: float, step: int) -> float:
        if step < self.warmup_iter:     return 1.0
        if dice_val < 0.90:             return 0.0
        if dice_val >= 0.92:            return 1.0
        frac = (0.92 - dice_val) / 100
        return math.exp(-self.kappa_lambda * (step - self.warmup_iter) * frac)

    def __call__(self,
                 mask_pred: torch.Tensor,
                 mask_gt:   torch.Tensor,
                 step:      int):
        zero = mask_pred.new_tensor(0.0)

        # ── normalize to [0,1]
        p_min, p_max = mask_pred.min(), mask_pred.max()
        p = (mask_pred - p_min) / (p_max - p_min + 1e-6)
        g = (mask_gt     - mask_gt.min()) / (mask_gt.max()   - mask_gt.min()   + 1e-6)

        # ── 1) KL loss exactly as in your old MaskShapeKLLoss
        x = p[p > 0]
        if x.numel() == 0:
            kl_loss = zero
        else:
            is_spike = x > self.spike_thr
            log_p    = torch.empty_like(x)
            # spike mass
            log_p[is_spike] = math.log(self.pi_star + self.eps)
            # Beta tail
            tail = ~is_spike
            if tail.any():
                log_p[tail] = (
                    math.log(1 - self.pi_star + self.eps)
                    + self._beta_pdf_log(x[tail])
                )
            kl = -log_p.mean()
            kl_loss = self._kl_anneal(step) * kl

        # ── 2) Dice & κ
        dice_val  = self._soft_dice(p, g, self.eps)
        kappa     = self._kappa_from_dice(float(dice_val), step)
        dice_loss = (
            self.dice_weight * (1 - dice_val)
            if step >= self.warmup_iter else zero
        )

        # ── 3) one-sided exponential non-zero fraction penalty
        soft_nz = torch.sigmoid((p - self.nz_thresh) / self.nz_soft_width)
        obs_nz  = soft_nz.mean()
        diff    = F.relu(self.target_nz_frac - obs_nz)
        nz_loss = self.lambda_nz * (torch.exp(self.nz_exp_factor * diff) - 1.0)

        return kl_loss, dice_loss, nz_loss, kappa

class MaskShapeKLLoss:
    r"""
    KL  =  –  E_x∼p̂  [ log  p*(x) ]           (empirical → template)

          with                           ⎧ π*                          if x≈1
                p*(x) =  π* δ(x-1)  +    ⎨
                                         ⎩ (1-π*) · Beta(x;α*,β*)      otherwise

    • Works on **non-zero** pixels only.
    • Uses a spike-threshold (≈0.999) to decide “exactly 1”.
    • Fully differentiable w.r.t. `mask_pred` (the ASG output); the
      template parameters are fixed constants.

    You still get the usual annealing β(step) like in your old class.
    """

    def __init__(
        self,
        pi_star: float,      # template spike probability
        alpha_star: float,   # template Beta α
        beta_star: float,    # template Beta β
        spike_thresh: float = 0.999,
        # β-anneal:
        beta_start: float = 1e-3,
        beta_final: float = 1e-4,
        t_start:    int   = 8000,
        t_end:      int   = 50_000,
        eps: float = 1e-6
    ):
        # template mixture params
        self.pi_star     = float(pi_star)
        self.alpha_star  = float(alpha_star)
        self.beta_star   = float(beta_star)
        self.spike_thr   = spike_thresh
        self.eps         = eps
        # pre-compute log-Beta(α*,β*)
        self.log_B_star  = (
            gammaln(torch.tensor(alpha_star))
          + gammaln(torch.tensor(beta_star))
          - gammaln(torch.tensor(alpha_star + beta_star))
        ).item()

        # anneal coeff
        self.beta_start  = beta_start
        self.beta_final  = beta_final
        self.t_start     = t_start
        self.t_end       = max(t_end, t_start + 1)

    # ─────────────────────────────────────────────
    def _beta_pdf_log(self, x: torch.Tensor) -> torch.Tensor:
        # log  BetaPDF(x; α*,β*)
        a, b = self.alpha_star, self.beta_star
        return ((a - 1.0) * torch.log(torch.clamp(x, self.eps, 1.0))
              + (b - 1.0) * torch.log(torch.clamp(1.0 - x, self.eps, 1.0))
              - self.log_B_star)

    def _anneal(self, step: int) -> float:
        if step < self.t_start:  return self.beta_start
        if step >= self.t_end:   return self.beta_final
        t = (step - self.t_start) / (self.t_end - self.t_start)
        return (1.0 - t) * self.beta_start + t * self.beta_final
    # ─────────────────────────────────────────────
    def __call__(self, mask_pred: torch.Tensor, step: int) -> torch.Tensor:
        """
        mask_pred : (B, H, W) or (H, W) float in [0,1] – rendered occupancy mask
        step      : global training iteration  (int)
        Returns   : scalar KL loss
        """
        x = mask_pred[mask_pred > 0]       # non-zero only
        if x.numel() == 0:                 # empty mask → no loss
            return mask_pred.new_tensor(0.0)

        is_spike = x >= self.spike_thr     # “δ at 1”
        tail     = ~is_spike

        # log p*(x) for every sample
        log_p = torch.empty_like(x)

        # spike part: probability mass π*
        log_p[is_spike] = math.log(self.pi_star + self.eps)

        # tail part: (1-π*) · BetaPDF
        if tail.any():
            log_tail = self._beta_pdf_log(x[tail]) + math.log(1.0 - self.pi_star + self.eps)
            log_p[tail] = log_tail

        # Empirical KL  =  –mean log p*
        kl = -log_p.mean()

        return self._anneal(step) * kl


def _bchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:          # (C,H,W)  →  (1,C,H,W)
        return x.unsqueeze(0)
    if x.dim() == 4:          # already batched
        return x
    raise ValueError(f"Expected 3- or 4-D tensor, got {x.shape}")


def rgb_to_luminance(x: torch.Tensor) -> torch.Tensor:
    # x is (B,C,H,W)
    return 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]


def linear_to_lpips(
        x_lin: torch.Tensor,
        eps: float = 1e-6               # ← ε-clamp here
) -> torch.Tensor:
    """
    Linear-RGB [0,1]  →  sRGB (IEC 61966-2-1) in the range [-1,1] expected by LPIPS.
    Adding eps > 0 guarantees x**(1/2.4) has a finite gradient.
    """
    a, k0 = 0.055, 0.0031308           # forward TF knot

    # 1) clip and ε-clamp
    x = x_lin.clamp(min=eps, max=1.0)

    # 2) piece-wise transfer function
    x_srgb = torch.where(
        x <= k0,
        12.92 * x,
        (1 + a) * x.pow(1 / 2.4) - a
    )

    # 3) scale to [-1, 1] for LPIPS
    return x_srgb * 2.0 - 1.0

# ---------------- main class ------------------------------------------
class HairDetailLoss:
    """
    Composite loss with three phases
      A) Huber                       (coarse)
      B) L1 + MS-SSIM + Sobel-Y      (detail)
      C) L1 + MS-SSIM + Sobel-Y + LPIPS (polish)

    Optional per-pixel gate_map: pixel‐trust g(u) ∈ (0,1].
    All magnitude‐type residuals (Huber, L1, Sobel) are weighted by g(u),
    and SSIM/LPIPS by g(u) as well.
    """

    def __init__(
        self,
        warmup_iters: int = 6_000,
        fade_iters:   int = 20_000,
        polish_iters: int = 5_000,
        huber_delta:  float = 0.10,
        w_ssim_final: float = 0.00,
        w_grad_final: float = 0.00,
        w_lpips_final:float = 0.00,
        blur:         bool  = False,
        blur_sigma:   float = 1.0,
        device:       str   = "cuda"
    ):
        # phase boundaries
        self.warmup   = warmup_iters
        self.fade     = fade_iters
        self.polish   = polish_iters

        # weights & settings
        self.delta     = huber_delta
        self.w_ssim_f  = w_ssim_final
        self.w_grad_f  = w_grad_final
        self.w_lpips_f = w_lpips_final
        self.blur       = blur
        self.blur_sigma = blur_sigma

        # LPIPS network (frozen)
        self.lpips = lpips.LPIPS(net='vgg').to(device).eval()
        for p in self.lpips.parameters():
            p.requires_grad_(False)

    def _huber(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff   = x - y
        abs_e  = diff.abs()
        quad   = 0.5 * diff ** 2
        linear = self.delta * (abs_e - 0.5 * self.delta)
        return torch.where(abs_e <= self.delta, quad, linear)

    def __call__(
        self,
        pred     : torch.Tensor,        # (B,3,H,W) or (3,H,W)
        tgt      : torch.Tensor,        # same shape as pred
        step     : int,
        gate_map : torch.Tensor = None  # (1,H,W) or (H,W)
    ) -> torch.Tensor:

        # ensure BCHW
        pred4, tgt4 = _bchw(pred), _bchw(tgt)

        # build per-pixel gate g(u)
        if gate_map is None:
            g_map = torch.ones_like(pred4[:, :1])  # (B,1,H,W)
        else:
            g_map = _bchw(gate_map)

        # phase‐dependent global weights
        if step < self.warmup:               # Phase A
            w_hu, w_l1, w_ss, w_gr, w_lp = 1, 0, 0, 0, 0
        elif step < self.warmup + self.fade: # Phase B
            t = (step - self.warmup) / self.fade
            w_hu = 1 - t;  w_l1 = t
            w_ss  = t * self.w_ssim_f
            w_gr  = t * self.w_grad_f
            w_lp  = 0
        else:                                # Phase C
            w_hu = 0; w_l1 = 1
            w_ss  = self.w_ssim_f
            w_gr  = self.w_grad_f
            if self.polish > 0:
                t_lp = min(1.0, (step - (self.warmup + self.fade)) / self.polish)
                w_lp = t_lp * self.w_lpips_f
            else:
                w_lp = self.w_lpips_f

        loss = 0.0

        # Phase A: Huber
        if w_hu:
            hub = self._huber(pred4, tgt4)               # (B,3,H,W)
            loss += w_hu * (hub * g_map).sum() / (g_map.sum() + 1e-6)

        # Phase B,C: L1
        if w_l1:
            l1 = (pred4 - tgt4).abs()                    # (B,3,H,W)
            loss += w_l1 * (l1 * g_map).sum() / (g_map.sum() + 1e-6)

        # SSIM (scale‐free) — weight by gate only
        if w_ss:
            ssim_map = 1 - ms_ssim(
                pred4, tgt4,
                window_size=11, max_val=1.0,
                reduction='none'
            )                                           # (B,1,H,W)
            loss += w_ss * (ssim_map * g_map).sum() / (g_map.sum() + 1e-6)

        # Sobel‐Y (detail) — magnitude‐type so weight by gate
        if w_gr:
            lum_p = rgb_to_luminance(pred4)
            lum_t = rgb_to_luminance(tgt4)
            if self.blur:
                k = int(2 * math.ceil(2 * self.blur_sigma) + 1)
                lum_t = gaussian_blur2d(
                    lum_t, (k, k),
                    (self.blur_sigma, self.blur_sigma)
                )
            grad = (sobel(lum_p) - sobel(lum_t)).abs()  # (B,1,H,W)
            loss += w_gr * (grad * g_map).sum() / (g_map.sum() + 1e-6)

        # LPIPS (polish)
        # if w_lp:
        #     lp = self.lpips(
        #         linear_to_lpips(pred4), linear_to_lpips(tgt4)
        #     )                                          # (B,1,1,1)
        #     # Gate LPIPS by the average gate: if gate_map is zero, this term is zero
        #     loss += w_lp * lp.mean()

        return loss

def sdf_contain_and_flow(
        nphm_grid,
        gaussians,
        strand_vertices,           # (S,V,3) canonical strands
        outside_tol = 0.04,       # 3 mm: allowed outward slack
        tau         = 0.4,         # |g·t| tolerance (~66°)
        k_radius    = 2.8,         # 95 % mass shell
        chunk_size  = 2_000_000,
        phys_scale  = 1.0,
):
    """Returns (loss_contain, loss_flow)."""

    sid     = gaussians._strand_id
    centers, tangents = gaussians.world_centers_and_tangents(strand_vertices)
    scales            = gaussians.get_scaling[:len(sid)]
    sigma_perp, sigma_par = scales[:, 0], scales[:, 1]

    M        = centers.shape[0]
    device   = centers.device
    depth_all = torch.empty(M, device=device)
    dot_pen   = torch.empty(M, device=device)   # (|g·t| - tau)+

    for s in range(0, M, chunk_size):
        e   = min(s + chunk_size, M)
        pts = centers[s:e].unsqueeze(0)
        dist, grad = nphm_grid.sample(pts, phys_scale=phys_scale)
        dist  = dist[0]                                # (N,)
        g     = torch.nn.functional.normalize(grad[0], dim=1, eps=1e-9)

        cos_a = (g * tangents[s:e]).sum(dim=1)
        num   = k_radius * sigma_perp[s:e] * sigma_par[s:e]
        denom = torch.sqrt(cos_a**2 * sigma_perp[s:e]**2 +
                           (1 - cos_a**2) * sigma_par[s:e]**2) + 1e-9
        r_star = num / denom

        depth_all[s:e] = dist - r_star                # (+) outside
        dot_pen  [s:e] = torch.clamp(cos_a.abs() - tau, min=0.0)

    # -------- containment: only if depth > outside_tol ---------------
    loss_contain = torch.clamp(depth_all - outside_tol,
                               min=0.0).pow(2).mean()

    # -------- flow: only if depth ≤ 0 (inside or on) -----------------
    inside_mask = depth_all <= 0.0
    loss_flow = (dot_pen[inside_mask]**2).mean() if inside_mask.any() \
                else depth_all.new_tensor(0.0)

    return loss_contain, loss_flow

def sdf_containment_loss(
        nphm_grid,           # NPHMOctree
        gaussians,           # GaussianPerm
        strand_vertices,     # (S,V,3) canonical strands
        k_radius    = 2.8,   # 95 % mass shell
        chunk_size  = 200_000,
        phys_scale  = 1.0,   # 1 ⇒ canonical metres
):
    """
    Penalises Gaussians that lie OUTSIDE the SDF zero-surface.
    Returns a single scalar loss with gradients.
    """

    sid     = gaussians._strand_id
    centers, tangents = gaussians.world_centers_and_tangents(strand_vertices)
    scales           = gaussians.get_scaling[:len(sid)]
    sigma_perp, sigma_par = scales[:, 0], scales[:, 1]

    M        = centers.shape[0]
    device   = centers.device
    depth_all = torch.empty(M, device=device)

    for s in range(0, M, chunk_size):
        e    = min(s + chunk_size, M)
        pts  = centers[s:e].unsqueeze(0)                 # (1,N,3)
        dist, grad = nphm_grid.sample(pts, phys_scale=phys_scale)
        dist   = dist[0]                                 # (N,)
        normal = torch.nn.functional.normalize(grad[0], dim=1, eps=1e-9)

        cos_a = (normal * tangents[s:e]).sum(dim=1)
        num   = k_radius * sigma_perp[s:e] * sigma_par[s:e]
        denom = torch.sqrt(cos_a**2 * sigma_perp[s:e]**2 +
                           (1 - cos_a**2) * sigma_par[s:e]**2) + 1e-9
        r_star = num / denom

        depth_all[s:e] = dist - r_star                   # +ve ⇒ outside

    loss_contain = torch.clamp(depth_all, min=0.0).pow(2).mean()
    return loss_contain

def sdf_hair_band_loss(
        nphm_grid,            # NPHMOctree
        gaussians,            # GaussianPerm
        strand_vertices,      # (S,V,3) canonical strands
        eps_band     = 0.004, # 4 mm half-width of the “free” zone
        k_radius     = 2.8,   # 95 % mass shell
        chunk_size   = 200_000,
        phys_scale   = 1.0,
):
    """
    Returns a single scalar loss  L_band  with gradients.
    Inside |depth| <= eps_band there is *no* penalty.

       depth_i = dist_i - r*_i
       L_band  = mean( clamp(|depth_i| - eps_band, 0)² )
    """

    sid     = gaussians._strand_id

    # ---------- centers & tangents from canonical strands --------------
    centers, tangents = gaussians.world_centers_and_tangents(strand_vertices)
    scales            = gaussians.get_scaling[:len(sid)]           # (M,3)
    sigma_perp, sigma_par = scales[:, 0], scales[:, 1]

    M        = centers.shape[0]
    device   = centers.device
    depth_all = torch.empty(M, device=device)

    # ---------- chunked SDF queries ------------------------------------
    for s in range(0, M, chunk_size):
        e    = min(s + chunk_size, M)
        pts  = centers[s:e].unsqueeze(0)                 # (1,N,3)
        dist, grad = nphm_grid.sample(pts, phys_scale=phys_scale)
        dist   = dist[0]                                 # (N,)
        normal = F.normalize(grad[0], dim=1, eps=1e-9)   # (N,3)

        # anisotropic support radius
        cos_a = (normal * tangents[s:e]).sum(dim=1)
        num   = k_radius * sigma_perp[s:e] * sigma_par[s:e]
        denom = torch.sqrt(cos_a**2 * sigma_perp[s:e]**2 +
                           (1 - cos_a**2) * sigma_par[s:e]**2) + 1e-9
        r_star = num / denom

        depth_all[s:e] = dist - r_star                   # signed depth

    # ---------- band-pass hinge ----------------------------------------
    loss_band = torch.clamp(depth_all.abs() - eps_band,
                            min=0.0).pow(2).mean()
    return loss_band

def sdf_hair_losses(
        nphm_grid,            # NPHMOctree
        gaussians,            # GaussianPerm
        strand_vertices,      # (S,V,3) canonical strands
        k_radius      = 2.8,  # 95 % mass shell
        tol_inside    = 0.01,  # no inward tolerance
        outside_limit = 0.060, # 3 cm leash (set as you like)
        chunk_size    = 2000_000,
        phys_scale    = 1.00,  # canonical metres
):
    sid     = gaussians._strand_id

    """
    Returns (loss_in, loss_out) – both keep gradients.
    """
    # --- centers & tangents computed fresh from canonical strands ----
    centers, tangents = gaussians.world_centers_and_tangents(
                            strand_vertices)

    scales      = gaussians.get_scaling[:len(sid)]
    sigma_perp  = scales[:, 0]
    sigma_par   = scales[:, 1]

    M = centers.shape[0]
    device = centers.device
    depth_all = torch.empty(M, device=device)

    # ------------- chunked SDF queries -------------------------------
    for s in range(0, M, chunk_size):
        e = min(s + chunk_size, M)
        pts = centers[s:e].unsqueeze(0)              # (1,N,3)
        dist, grad = nphm_grid.sample(pts,
                                      phys_scale=phys_scale)
        dist = dist[0]                               # (N,)
        grad = grad[0]
        normal = torch.nn.functional.normalize(grad, dim=1, eps=1e-9)

        cos_ang = (normal * tangents[s:e]).sum(dim=1)
        num   = k_radius * sigma_perp[s:e] * sigma_par[s:e]
        denom = torch.sqrt(cos_ang**2 * sigma_perp[s:e]**2 +
                           (1-cos_ang**2) * sigma_par[s:e]**2) + 1e-9
        r_star = num / denom
        depth_all[s:e] = dist - r_star               # signed depth

    # ---------------- hinge losses -----------------------------------
    loss_in  = torch.clamp(-depth_all + tol_inside,
                           min=0.0).pow(2).mean()
    loss_out = torch.clamp( depth_all - outside_limit,
                           min=0.0).pow(2).mean()

    return loss_in, loss_out

def filtered_opacity_penalty(gaussians):
    """
    Penalize Gaussians whose *effective* opacity (after the 3D mip filter)
    falls below `thresh`.
    """
    # (M,1) tensor of α̃ = α * coef
    sid     = gaussians._strand_id

    eff_opacity = gaussians.get_opacity_with_3D_filter[:len(sid)]
    
    return ((1.0 - eff_opacity)**2).mean()

# predefine on CPU; we’ll move to the right device at runtime
_sobel_x = torch.tensor([[[[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]]]], dtype=torch.float32)
_sobel_y = _sobel_x.transpose(2,3)

def sobel_loss(render, gt, hair_mask=None, reduction="mean"):
    """
    render, gt: either [3,H,W] or [1,3,H,W] in [0..1]
    hair_mask: [H,W] or [1,H,W] or [1,1,H,W]
    """
    # --- make batch+channel dims consistent ---
    if render.dim() == 3:
        render = render.unsqueeze(0)    # [1,3,H,W]
    if gt.dim()     == 3:
        gt     = gt.unsqueeze(0)
    B, C, H, W = render.shape

    # --- convert to grayscale luminance ---
    # you can tweak these weights if you like
    r = render[:,0:1]
    g = render[:,1:2]
    b = render[:,2:3]
    render_gray = 0.299*r + 0.587*g + 0.114*b  # [1,1,H,W]

    r = gt[:,0:1]
    g = gt[:,1:2]
    b = gt[:,2:3]
    gt_gray     = 0.299*r + 0.587*g + 0.114*b

    # --- move kernels to the right device/dtype ---
    sobel_x = _sobel_x.to(render_gray.device).to(render_gray.dtype)
    sobel_y = _sobel_y.to(render_gray.device).to(render_gray.dtype)

    # --- convolve ---
    rx = F.conv2d(render_gray, sobel_x, padding=1)  # [1,1,H,W]
    ry = F.conv2d(render_gray, sobel_y, padding=1)
    gx = F.conv2d(gt_gray,     sobel_x, padding=1)
    gy = F.conv2d(gt_gray,     sobel_y, padding=1)

    # --- gradient‐difference ‖G(render) − G(gt)‖₁ ---
    diff = torch.abs(rx - gx) + torch.abs(ry - gy)   # [1,1,H,W]

    # --- optionally mask to hair pixels ---
    if hair_mask is not None:
        # bring mask to [1,1,H,W]
        m = hair_mask
        if m.dim() == 2:
            m = m.unsqueeze(0).unsqueeze(0)
        elif m.dim() == 3:
            m = m.unsqueeze(1)
        diff = diff * m

    # --- reduce ---
    if reduction == "mean":
        return diff.mean()
    elif reduction == "sum":
        return diff.sum()
    else:
        return diff

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
    sid     = gaussians._strand_id
    xyz = gaussians.get_xyz[:len(sid)]                         # (N,3)
    uv  = project_to_screen(
            xyz,
            viewpoint_cam.projection_matrix[:3, :3].unsqueeze(0),
            viewpoint_cam.w2c.unsqueeze(0)
          )[0].long()                               # (N,2) long

    u, v = uv[:, 0], uv[:, 1]
    in_view = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, in_view          # filtered u,v and mask


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

import torch, math, random
import torch.nn.functional as F
import matplotlib
import matplotlib.colors as mcolors
import numpy as np

# ─── tiny util ─────────────────────────────
def _angle_to_rgb(dx, dy):
    hue = (torch.atan2(dy, dx) + math.pi) / (2 * math.pi)
    rgb = torch.tensor(mcolors.hsv_to_rgb(
            torch.stack([hue, torch.ones_like(hue), torch.ones_like(hue)], 0)
                  .permute(1, 2, 0).cpu().numpy()), device=dx.device)
    return rgb.permute(2, 0, 1)          # (3,H,W)

def _draw_line(canvas, x0, y0, x1, y1, color):
    H, W = canvas.shape[1:]
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        if 0 <= x0 < W and 0 <= y0 < H:
            canvas[0, y0, x0] = color[0]
            canvas[1, y0, x0] = color[1]
            canvas[2, y0, x0] = color[2]
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy; x0 += sx
        if e2 < dx:
            err += dx; y0 += sy

# ─── debug-capable loss ────────────────────
def orientation_loss_v2_debug(
        viewpoint_cam,
        gaussians,
        hair_mask: torch.Tensor,      # (1,H,W)
        orient_map:  torch.Tensor,    # (C,H,W) or (H,W,C)
        occ_mask: torch.Tensor,
        mask_thresh: float = 0.5,
        debug: bool = False,
        max_arrows: int = 300):

    device = gaussians.get_xyz.device
    sid     = gaussians._strand_id
    occ_mask = occ_mask[:len(sid)]

    # channel split ------------------------------------------------------
    if orient_map.dim() == 3 and orient_map.size(0) == 3:      # (C,H,W)
        m, g, b = orient_map[0], orient_map[1], orient_map[2]
    elif orient_map.dim() == 3 and orient_map.size(2) == 3:    # (H,W,C)
        m, g, b = orient_map[..., 0], orient_map[..., 1], orient_map[..., 2]
    else:
        return (torch.tensor(0., device=device), None) if debug else torch.tensor(0., device=device)
    
    dx = b * 2.0 - 1.0
    dy = g * 2.0 - 1.0
    mask_pix = m > mask_thresh
    H, W = m.shape[-2:]

    _, hm, wm = hair_mask.shape
    if hm != H:
        hair_mask = F.interpolate(hair_mask.unsqueeze(1),                     
                                size=(H, W),                         
                                mode='bilinear', align_corners=False)[0]

    # project ------------------------------------------------------------
    u, v, in_view = _project_gaussians_to_uv(gaussians, viewpoint_cam, H, W)
    in_view = in_view & occ_mask
    u = u[in_view]
    v = v[in_view]

    if u.numel() == 0:
        return (torch.tensor(0., device=device), None) if debug else torch.tensor(0., device=device)
    idx_g = in_view.nonzero(as_tuple=False).squeeze(1)

    on_hair = mask_pix[v, u]
    if on_hair.sum() == 0:
        return (torch.tensor(0., device=device), None) if debug else torch.tensor(0., device=device)

    u, v  = u[on_hair], v[on_hair]
    idx_g = idx_g[on_hair]

    # gt / pred dirs -----------------------------------------------------
    gt_vec2d = F.normalize(torch.stack([dx[v, u], dy[v, u]], -1), dim=-1)
    gt_vec2d *= -1

    R_local  = quaternion_to_rotation_matrix(gaussians.get_rotation[:len(sid)])
    y_world  = R_local[:, :, 1]
    dirs_cam = convert_normal_to_camera_space(
                  y_world[idx_g],
                  viewpoint_cam.w2c[:3, :3],
                  viewpoint_cam.projection_matrix[:3, :3])[:, :2]
    dirs_2d  = F.normalize(dirs_cam, dim=-1)

    # cosine loss --------------------------------------------------------
    dot = (dirs_2d * gt_vec2d).sum(-1).clamp(-1., 1.)

    # now collapse opposite directions by taking abs()
    inv_dot = 1.0 - dot.abs()
    loss = inv_dot.mean()

    # scales   = gaussians.get_scaling
    # weight   = torch.exp(-(scales[idx_g, 0] / scales[idx_g, 1]).clamp_min(1e-6))
    # loss = (weight * inv_dot).mean()

    # fast exit if no debug ---------------------------------------------
    if not debug:
        return loss

    # ---- build overlay -------------------------------------------------
    canvas = torch.zeros(3, H, W, device=device)
    canvas[0] = 0.2; canvas[1] = 0.1; canvas[2] = 0.3   # purple bg
    canvas = canvas * (1 - hair_mask[0]) + hair_mask[0] * 0.15

    orient_rgb = _angle_to_rgb(dx, dy) * hair_mask[0]
    canvas = (canvas + orient_rgb).clamp(0, 1)

    # white dots: vectorised assignment
    canvas[:, v.long(), u.long()] = 1.0

    # arrow subset -------------------------------------------------------
    M = u.numel()
    if M > max_arrows:
        keep_idx = torch.randperm(M, device=device)[:max_arrows]
        u_draw   = u[keep_idx]; v_draw = v[keep_idx]
        g_sub    = gt_vec2d[keep_idx]; p_sub = dirs_2d[keep_idx]
    else:
        u_draw, v_draw, g_sub, p_sub = u, v, gt_vec2d, dirs_2d

    step = 4
    for uu_t, vv_t, gvec, pvec in zip(u_draw, v_draw, g_sub, p_sub):
        uu, vv = int(uu_t.item()), int(vv_t.item())
        # GT arrow (green)
        _draw_line(canvas, uu, vv,
                   int(round(uu + gvec[0].item()*step)),
                   int(round(vv + gvec[1].item()*step)),
                   color=(0,1,0))
        # Pred arrow (cyan)
        _draw_line(canvas, uu, vv,
                   int(round(uu + pvec[0].item()*step)),
                   int(round(vv + pvec[1].item()*step)),
                   color=(0,1,1))

    debug_np = (canvas.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    return loss, debug_np

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

def neighbour_orientation_loss(
    strand_points: torch.Tensor,
    neighbor_idx: torch.LongTensor,
    k: int = 6
) -> torch.Tensor:
    """
    Encourage neighbouring strands to have similar orientations,
    using a precomputed neighbour‐list (no O(S^2) cdist).
    Args:
        strand_points: (S, V, 3) tensor of strand vertices in scene space.
        neighbor_idx:  (S, K_MAX) LongTensor of neighbour‐indices per strand.
        k:             how many of those precomputed neighbours to use.
    Returns:
        Scalar loss.
    """
    device = strand_points.device
    S = strand_points.shape[0]
    if S < 2:
        return torch.tensor(0.0, device=device)

    # effective neighbour count
    k_eff = min(k, neighbor_idx.size(1))

    # 1) extract tangents at root
    tangents = strand_points[:, 1] - strand_points[:, 0]            # (S,3)
    tangents = F.normalize(tangents, dim=1, eps=1e-6)               # (S,3)

    # 2) gather each strand's k_eff neighbours
    idx_knn = neighbor_idx[:, :k_eff]                               # (S, k_eff)
    nbr_tangs = tangents[idx_knn]                                   # (S, k_eff, 3)

    # 3) cosine similarity
    cos_sim = (tangents.unsqueeze(1) * nbr_tangs).sum(dim=2)        # (S, k_eff)

    # 4) hinge‐style penalty: mean(1 - cos_sim)
    loss = (1.0 - cos_sim).clamp(min=0).mean()
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
    sid     = gaussians._strand_id

    device = gaussians.get_xyz.device
    H, W = hair_mask.shape[1:]

    # 1) project every Gaussian’s center into pixel coords
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
    opacity = gaussians.get_opacity[:len(sid)].squeeze(-1)   # (N,)
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
    sid     = gaussians._strand_id

    # --- build edges & unit‐dirs ---
    edges = strands[:, 1:, :] - strands[:, :-1, :]        # (S, N-1, 3)
    M = edges.numel() // 3
    edge_dirs = edges.reshape(M, 3)
    edge_lens = edge_dirs.norm(dim=1, keepdim=True).clamp(min=1e-6)
    dirs = edge_dirs / edge_lens                            # (M,3)

    # --- get live quaternions and normalize ---
    quats = gaussians.get_rotation[:len(sid)]  # (M,4)
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
    sid     = gaussians._strand_id
    edges = strands[:, 1:, :] - strands[:, :-1, :]
    M = edges.numel() // 3

    # --- get live scales ---
    scales = gaussians.scaling_activation(gaussians._scaling_base[:len(sid)])  # (M,2)
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
    sid     = gaussians._strand_id
    # --- build edges & true lengths ---
    edges = strands[:, 1:, :] - strands[:, :-1, :]
    M = edges.numel() // 3
    edge_dirs = edges.reshape(M, 3)
    edge_lens = edge_dirs.norm(dim=1, keepdim=True)        # (M,1)

    # --- get live y‐scale ---
    scales = gaussians.scaling_activation(gaussians._scaling_base[:len(sid)])  # (M,2)
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
    D_pred: torch.Tensor,      # (H,W) inverted depth, 0 = bg
    D_gt:   torch.Tensor,      # (H,W) GT depth [0,1] (near→0, far→1)
    mask:   torch.Tensor,      # (H,W) soft mask 0-1
    shrink_thr:float = 0.9,
    q_lo: float = 0.4, q_hi: float = 1.0,
    halo_kernel: int = 11,
    mode: str = "l2"):

    device = D_pred.device

    # A) build *core* mask (eroded) via average-pool  --------------------
    if mask.dim() == 3:
        m = (mask > shrink_thr).any(0).float()   # (H,W)
    else:
        m = (mask > shrink_thr).float()

    pad = halo_kernel // 2
    m_core = (F.avg_pool2d(m[None,None], halo_kernel, 1, pad)[0,0] > 0.99)
    if m_core.sum() < 10:
        return torch.tensor(0., device=device), torch.zeros_like(D_pred)

    # B) robust percentiles on core mask --------------------------------
    vals = D_pred[m_core & (D_pred != 0)]
    near = torch.quantile(vals, q_lo)
    far  = torch.quantile(vals, q_hi)

    D_norm = (D_pred - near) / (far - near + 1e-6)
    D_norm = 1.0 - D_norm.clamp(0,1)             # invert (near=0)

    # C) loss over *weighted* region ------------------------------------
    #   core pixels weight 1, halo fades linearly with avg_pool
    halo_conf = F.avg_pool2d(m[None,None], halo_kernel, 1, pad)[0,0]
    weights = halo_conf.clamp(0,1)

    if mode.lower() == "l2":
        diff = (D_norm - D_gt)**2
    elif mode.lower() == "l1":
        diff = torch.abs(D_norm - D_gt)
    else:
        raise ValueError("mode must be 'l1' or 'l2'")

    loss = (weights * diff).sum() / weights.sum()
    return loss, D_norm

def aligned_depth_loss_old(
    D_pred: torch.Tensor,      # (H,W) raw predicted depth (world units)
    D_gt:   torch.Tensor,      # (H,W) ground-truth normalized depth [0,1]
    mask:   torch.Tensor,      # (H,W) hair mask ∈{0,1}
    low_pct:  float = 0.04,     # lower Windsor percentile
    high_pct: float = 0.98,    # upper Windsor percentile
    mode:     str   = "l2",    # "l2" or "l1"
    eps:      float = 1e-6
):
    """
    1) Clip D_pred[mask] to its [low_pct,high_pct] percentiles.
    2) Rescale clipped D_pred to [0,1].
    3) Compute loss vs D_gt over mask.

    Returns:
        loss      : scalar tensor
        D_normal  : (H,W) tensor in [0,1] for visualization/debug
    """
    device = D_pred.device

    # --- 1. extract only hair-pixels and compute percentiles ---
    vals = D_pred[mask == 1]
    vals = vals[vals != 0.0]
    if vals.numel() < 2:
        # fallback to zero loss if no hair pixels
        D_norm = torch.zeros_like(D_pred)
        return torch.tensor(0.0, device=device), D_norm

    p_lo = vals.quantile(low_pct)
    p_hi = vals.max()
    min_locs = D_pred == 0.0

    # --- 2. clip & normalize entire map ---
    D_clip = torch.clamp(D_pred, p_lo, p_hi)
    D_norm = (D_clip - p_lo) / ( (p_hi - p_lo) + eps )
    D_norm = 1 - D_norm.clamp(0.0, 1.0)
    D_norm[min_locs] = 0.0

    # --- 3. compute error on hair region ---
    mask_h = mask > 0.5
    if mode.lower() == "l2":
        diff = D_norm[mask_h] - D_gt[mask_h]
        loss = (diff * diff).mean()
    elif mode.lower() == "l1":
        loss = torch.abs(D_norm[mask_h] - D_gt[mask_h]).mean()
    else:
        raise ValueError("mode must be 'l2' or 'l1'")

    return loss, D_norm


def head_collision_loss(
    strand_pts: torch.Tensor,     # (S, V, 3)
    head_pts:   torch.Tensor,     # (H, 3)
    head_nmls:  torch.Tensor,     # (H, 3)
    margin:     float = 0.001,    # metres
    reduction:  str = "mean"      # "none"|"sum"|"mean"
) -> torch.Tensor:
    """
    strand_pts : (S,V,3) hair‐vertex positions
    head_pts   : (H,3)   sampled head‐point positions
    head_nmls  : (H,3)   corresponding unit normals (pointing outward)
    """
    # 1) flatten hair points → (1, N, 3)
    N = strand_pts.shape[0] * strand_pts.shape[1]
    pts = strand_pts.reshape(1, N, 3)

    # 2) head pointcloud → (1, H, 3)
    head = head_pts.unsqueeze(0)

    # 3) knn search (returns squared dists & indices)
    knn = knn_points(pts, head, K=1, return_nn=False)
    idx = knn.idx.squeeze(0).squeeze(-1)         # (N,)

    # 4) gather closest head point + normal
    closest = head_pts[idx]                      # (N,3)
    nrm     = head_nmls[idx]                     # (N,3)

    # 5) signed distance = dot( p - closest, normal )
    delta = pts.squeeze(0) - closest             # (N,3)
    d_signed = (delta * nrm).sum(dim=1)          # (N,)

    # 6) hinge‐violation: margin – d_signed
    viol = F.relu(margin - d_signed)             # (N,)

    loss = viol.pow(2)                           # quadratic
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction {reduction!r}")

def gaussian_head_collision_loss(
    gaussians,                   # your GaussianPerm instance
    head_pts: torch.Tensor,      # (H,3) mesh.vertices as torch.Tensor
    head_nmls: torch.Tensor,     # (H,3) mesh.vertex_normals as torch.Tensor
    margin: float = 0.001,       # metres of clearance
    reduction: str = "mean"      # "none" | "sum" | "mean"
) -> torch.Tensor:
    """
    Ensure no Gaussian center lies inside the head mesh (or closer than margin).

    We take:
      p_i     = gaussians.get_xyz (M,3)
      head    = head_pts           (H,3)
      normals = head_nmls          (H,3)

    1) For each p_i, find nearest head point p_h via knn_points.
    2) Compute signed‐distance = (p_i - p_h) · n_h.
    3) Penalize margin - signed_distance if signed_distance < margin.
    """
    # 1) collect Gaussian centers → (1, M, 3)
    sid     = gaussians._strand_id
    pts = gaussians.get_xyz[:len(sid)].unsqueeze(0)     # (1, M, 3)

    # 2) head point‐cloud → (1, H, 3)
    head = head_pts.unsqueeze(0)             # (1, H, 3)

    # 3) KNN query for nearest head‐point
    knn = knn_points(pts, head, K=1, return_nn=False)
    idx = knn.idx.squeeze(0).squeeze(-1)     # (M,)

    # 4) gather the matched head‐points + normals
    closest = head_pts[idx]                  # (M,3)
    normals = head_nmls[idx]                 # (M,3)

    # 5) signed distance along the normal
    delta = pts.squeeze(0) - closest         # (M,3)
    d_signed = (delta * normals).sum(dim=1)  # (M,)

    # 6) hinge on margin
    viol = F.relu(margin - d_signed)         # (M,)
    loss = viol.pow(2)                       # (M,)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction!r}")

def strand_repulsion_loss(
    strand_pts: torch.Tensor,     # (S, V, 3) all strand vertices
    gaussians,                    # GaussianPerm instance (has _strand_radius, roots)
    k_strand:    int = 18,         # how many neighbour-strands to consider per strand
    step:        int = 1,         # subsample every `step` vertices
    safe_dist:   float = 0.000    # metres
) -> torch.Tensor:
    """
    Vectorized strand repulsion that only checks between strands whose roots
    are k_strand-nearest neighbours.

    Returns a scalar hinge loss.
    """
    sid     = gaussians._strand_id
    device = strand_pts.device
    S, V, _ = strand_pts.shape
    if S < 2:
        return torch.tensor(0.0, device=device)

    # 1) subsample each strand’s points
    sampled = strand_pts[:, ::step]          # (S, P_each, 3)
    P_each = sampled.shape[1]

    # 2) find k_strand nearest-neighbour strands by root positions
    roots = gaussians.roots                   # (S, 3)
    dist_roots = torch.cdist(roots, roots, p=2)  # (S, S)
    _, nbrs_all = dist_roots.topk(k=k_strand+1, dim=1, largest=False)
    nbrs = nbrs_all[:, 1:]                    # (S, k_strand)

    # 3) build all unordered (s < t) neighbour pairs
    s_idx = torch.arange(S, device=device).unsqueeze(1).expand(-1, k_strand)  # (S, k_strand)
    mask = nbrs > s_idx                       # only keep t > s
    pair_s = s_idx[mask]                      # (P,)
    pair_t = nbrs[mask]                       # (P,)
    P = pair_s.shape[0]
    if P == 0:
        return torch.tensor(0.0, device=device)

    # 4) gather sampled points for each pair
    pts_s = sampled[pair_s]  # (P, P_each, 3)
    pts_t = sampled[pair_t]  # (P, P_each, 3)

    # 5) compute batched squared distances: (P, P_each, P_each)
    d_mat = torch.cdist(pts_s, pts_t, p=2.0)  # (P, P_each, P_each)
    d2 = d_mat.pow(2)

    # 6) compute hinge threshold per pair: (r_s + r_t + safe_dist)^2 → (P,)
    r = gaussians._strand_radius[:len(sid)]  # (S,)
    r_s = r[pair_s]               # (P,)
    r_t = r[pair_t]               # (P,)
    thresh2 = (r_s + r_t + safe_dist).pow(2).view(P, 1, 1)

    # 7) hinge loss: ReLU(thresh2 - d2), mean over each (P_each × P_each) block
    loss_mat = torch.relu(thresh2 - d2)        # (P, P_each, P_each)
    loss_pairs = loss_mat.mean(dim=(1, 2))     # (P,)

    # 8) final scalar: mean over all P neighbour-pairs
    return loss_pairs.mean()

def gaussian_scale_regularization_loss(
    gaussians,
    max_ratio: float = 30.0,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Hinge‐style penalty if any Gaussian’s tangential scale σ∥
    exceeds max_ratio × its perpendicular scale σ⊥.

    Args:
        gaussians: your GaussianPerm instance
        max_ratio: allowed σ∥/σ⊥ before penalty kicks in
        eps:       small value to avoid division by zero
    Returns:
        scalar loss
    """
    # (M,3): (σ⊥, σ∥, σ⊥)
    sid     = gaussians._strand_id
    scales = gaussians.get_scaling[:len(sid)]
    perp   = scales[:, 0]              # (M,)
    para   = scales[:, 1]              # (M,)

    # current ratio
    cur_ratio = para / (perp + eps)    # (M,)

    # hinge: excess ratio
    excess = (cur_ratio - max_ratio).clamp(min=0.0)

    # quadratic penalty
    return (excess ** 2).mean()

def local_length_consistency_loss(
    gaussians,
    k: int = 6
) -> torch.Tensor:
    """
    Enforce that nearby strands (by root position) have similar total lengths,
    using a precomputed neighbour‐list (no O(S^2) cdist).
    """
    device = gaussians.roots.device
    S = gaussians.num_strands
    if S < 2:
        return torch.tensor(0.0, device=device)

    # effective neighbourhood size
    k_eff = min(k, gaussians.neighbor_idx.size(1))

    # get neighbour indices
    neighbors = gaussians.neighbor_idx[:, :k_eff]                   # (S, k_eff)

    # strand lengths
    lengths = gaussians.strand_length                               # (S,)

    # gather the neighbour‐lengths
    nbr_lengths = lengths[neighbors]                                # (S, k_eff)

    # squared differences
    diff = lengths.unsqueeze(1) - nbr_lengths                       # (S, k_eff)
    return (diff ** 2).mean()


def color_variance_loss_sh(gaussians) -> torch.Tensor:
    """
    Compute the mean variance of SH color features across Gaussians on each strand.
    """
    sid     = gaussians._strand_id

    feat_all = gaussians.get_features[:len(sid)]           # (M, 3, n_sh)
    M, _, n_sh = feat_all.shape
    D = n_sh * 3
    feat_flat = feat_all.view(M, D)             # (M, D)

    strand_id = gaussians._strand_id            # (M,)
    S = gaussians.num_strands
    device = feat_flat.device

    # Count Gaussians per strand
    counts = torch.bincount(strand_id, minlength=S).unsqueeze(1).float()  # (S,1)
    counts_clamped = counts.clamp_min(1.0)

    # Sum of features per strand: (S, D)
    sum_feats = torch.zeros((S, D), device=device)
    sum_feats = sum_feats.index_add(0, strand_id, feat_flat)

    # Sum of squares per strand: (S, D)
    sum_feats_sq = torch.zeros((S, D), device=device)
    sum_feats_sq = sum_feats_sq.index_add(0, strand_id, feat_flat * feat_flat)

    # Variance per feature: E[x^2] - (E[x])^2
    mean_feats = sum_feats / counts_clamped       # (S, D)
    mean_feats_sq = sum_feats_sq / counts_clamped # (S, D)
    var_feats = mean_feats_sq - mean_feats * mean_feats  # (S, D)

    # Collapse each strand's D-dimensional variance to a scalar, then average
    strand_var = var_feats.mean(dim=1)             # (S,)
    return strand_var.mean()

def asg_variance_loss(gaussians) -> torch.Tensor:
    """
    Mean per-strand variance of the 24-D ASG colour features.

    • get_asg_features() → (M, 24) tensor of per-Gaussian colour coefficients
    • _strand_id            → (M,)  LongTensor   mapping Gaussians → strand index
    • num_strands           → scalar S

    The loss is the average (over strands) of the average (over the 24 channels)
    variance within each strand.

        L = 1/S  ∑_s  ( 1/24 ∑_d  Var_s[ f_{⋅d} ] )
          = 1/S  ∑_s  (1/24) ∑_d (E[f²] – (E[f])²)_sd
    """
    # 1 ─ fetch data -------------------------------------------------------
    strand_id = gaussians._strand_id                  # (M,)
    asg_all   = gaussians.get_asg_features[strand_id]            # (M, 24)
    S         = gaussians.num_strands
    device    = asg_all.device
    D         = asg_all.shape[1]                      # 24

    # 2 ─ per-strand counts  (S,)
    counts = torch.bincount(strand_id, minlength=S).float().clamp_min_(1.0)

    # 3 ─ per-strand sum of features  (S, 24)
    sum_feat = torch.zeros(S, D, device=device)
    sum_feat.index_add_(0, strand_id, asg_all)

    # 4 ─ per-strand sum of squared features  (S, 24)
    sum_feat_sq = torch.zeros(S, D, device=device)
    sum_feat_sq.index_add_(0, strand_id, asg_all * asg_all)

    # 5 ─ means & mean-square  --------------------------------------------
    mean_feat    = sum_feat    / counts[:, None]      # (S, 24)
    mean_feat_sq = sum_feat_sq / counts[:, None]      # (S, 24)

    # 6 ─ per-strand, per-channel variance
    var = torch.clamp(mean_feat_sq - mean_feat * mean_feat, min=0.0)  # (S, 24)

    # 7 ─ average over channels, then over strands → scalar loss
    return var.mean()

def opacity_variance_loss(gaussians) -> torch.Tensor:
    """
    Compute the mean variance of per‐Gaussian opacity across all Gaussians on each strand,
    in a memory‐efficient way and clamped to ≥0.
    """
    sid     = gaussians._strand_id

    # 1) Fetch each Gaussian’s opacity as a 1-D tensor (M,)
    opac_all  = gaussians.get_opacity_with_3D_filter[:len(sid)].squeeze(-1)  # (M,)

    # 2) Strand‐ID mapping
    strand_id = gaussians._strand_id                               # (M,)
    S         = gaussians.num_strands
    device    = opac_all.device

    # 3) Count Gaussians per strand → (S,)
    counts = torch.bincount(strand_id, minlength=S).float()       # (S,)
    counts = counts.clamp_min(1.0)                                # avoid div by zero

    # 4) Sum of opacities per strand → (S,)
    sum_op    = torch.zeros(S, device=device)
    sum_op    = sum_op.index_add_(0, strand_id, opac_all)

    # 5) Sum of squared opacities per strand → (S,)
    sum_op_sq = torch.zeros(S, device=device)
    sum_op_sq = sum_op_sq.index_add_(0, strand_id, opac_all * opac_all)

    # 6) Compute per‐strand means
    mean_op    = sum_op    / counts   # (S,)
    mean_op_sq = sum_op_sq / counts   # (S,)

    # 7) Variance per strand and clamp to ≥0
    var_per_strand = mean_op_sq - mean_op * mean_op  # (S,)
    var_per_strand = torch.clamp(var_per_strand, min=0.0)

    # 8) Return the average variance across strands
    return var_per_strand.mean()

def triangle_scale_area_loss(gaussians,
                             beta: float = 2e6,
                             lam: float = 0.0) -> torch.Tensor:
    """
    Robust triangle-area loss with soft‐max outlier penalty.

    For each Gaussian, build the triangle whose vertices are:
      • P0      : center
      • P1      : P0 + 0.5 * sigma_para * tangent
      • P_close : projection of P1 onto the local strand

    area = 0.5 * ||P1−P0|| * ||P_close−P0||

    Loss = mean(area) + lam * (1/beta) * log_mean_exp(beta * area)

    where (1/beta) * log_mean_exp(beta * a) is a smooth approximation of max(a).
    """

    # 1) compute half‐scale area per Gaussian (as before) …
    sid     = gaussians._strand_id
    scales     = gaussians.get_scaling[:len(sid)]                 # (M,3)
    sigma_para = scales[:,1] * 0.5                     # (M,)

    P0 = gaussians.get_xyz[:len(sid)]                              # (M,3)

    R_local = quaternion_to_rotation_matrix(gaussians.get_rotation[:len(sid)])  # (M,3,3)
    tangent = R_local[:,:,1]                                         # (M,3)

    v  = tangent * sigma_para.unsqueeze(1)      # (M,3)
    P1 = P0 + v                                  # (M,3)

    # local‐segment projection (vectorized, O(M))
    s_param = gaussians.get_axial_weight[:len(sid)]
    strands = gaussians.strands                  # (S,V,3)
    S, V, _  = strands.shape

    idx = (s_param*(V-1)).floor().long().clamp(0, V-2)  # (M,)
    segs = torch.stack([idx,
                        (idx-1).clamp(0,V-2),
                        (idx+1).clamp(0,V-2)],
                       dim=1)                       # (M,3)
    flat = strands.view(-1,3)                         # (S*V,3)
    base = sid * V                                    # (M,)
    idx_flat = base.unsqueeze(1) + segs               # (M,3)
    A = flat[idx_flat]                                # (M,3,3)
    B = flat[idx_flat+1]                              # (M,3,3)

    AB  = B - A                                       # (M,3,3)
    AB2 = (AB*AB).sum(dim=2).clamp(min=1e-8)           # (M,3)
    AP  = P1.unsqueeze(1) - A                         # (M,3,3)
    t   = (AP*AB).sum(dim=2) / AB2                    # (M,3)
    t_cl= t.clamp(0,1).unsqueeze(2)                   # (M,3,1)
    P_proj = A + AB*t_cl                              # (M,3,3)

    d2    = ((P1.unsqueeze(1) - P_proj)**2).sum(dim=2)  # (M,3)
    best  = d2.argmin(dim=1)                           # (M,)
    P_close = P_proj[torch.arange(P_proj.size(0)), best]  # (M,3)

    u     = P_close - P0                               # (M,3)
    area  = 0.5 * v.norm(dim=1) * u.norm(dim=1)        # (M,)

    # # 2) soft‐max outlier penalty via log‐mean‐exp
    # #    log_mean_exp = log( mean(exp(beta*area)) )
    # lme  = (area * beta).exp().mean().log()
    # soft_max = lme / beta

    # 3) combined loss
    return (area ** 4).mean()