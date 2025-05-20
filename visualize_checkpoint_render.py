#!/usr/bin/env python3
# visualize_checkpoint_render.py
#
# Outputs:
#   render.mp4, segmentation.mp4, depth.mp4, comparison.mp4
# Comparison layout:
#   GT-masked | Pred-RGB | GT-Seg | Pred-Seg | GT-Depth | Pred-Depth | Orient

import os, math, argparse, pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import torch, imageio
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

from hair.hair_models          import Perm
from scene.gaussian_perm       import GaussianPerm
from scene                     import Scene_mica
from src.perm_deform_model     import PermDeformModel
from gaussian_renderer         import render
from arguments                 import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils       import to_image_np

# ───────────────── depth normalisation ──────────────────
def soft_percentile(x: torch.Tensor, mask: torch.Tensor,
                    q: float, sharp=20.0) -> torch.Tensor:
    vals = x[mask].view(-1)
    k = max(1, round(q * vals.numel()))
    scores = vals * sharp
    topk, _ = scores.topk(k)
    return (topk / sharp).logsumexp(0) - math.log(k)

def depth_norm_shrink(d_pred: torch.Tensor,
                      mask:  torch.Tensor,
                      thr:   float = 0.9,
                      q_lo:  float = 0.02,
                      q_hi:  float = 0.98) -> torch.Tensor:
    if mask.dim() == 3:
        hard = (mask > thr).any(dim=0)
    else:
        hard = mask > thr
    hard_f = hard.float().to(d_pred.device)

    valid = hard & (d_pred != 0)
    if valid.sum() < 10:
        return torch.zeros_like(d_pred)

    near = soft_percentile(d_pred, valid, q_lo)
    far  = soft_percentile(d_pred, valid, q_hi)
    d_norm = (d_pred - near) / (far - near + 1e-6)
    return (d_norm.clamp(0, 1) * hard_f)

def depth_to_rgb(d_pred: torch.Tensor,
                 mask:   torch.Tensor,
                 invert: bool = False) -> np.ndarray:
    d_vis = depth_norm_shrink(d_pred, mask)
    if invert:
        d_vis = 1.0 - d_vis
    rgb = plt.get_cmap("turbo")(d_vis.cpu().numpy())[..., :3]
    return (rgb * 255).astype(np.uint8)

# ───────────── orientation visualisation ───────────────
def orient_to_rgb(orient: torch.Tensor) -> np.ndarray:
    if orient.dim() == 3 and orient.size(0) == 3:          # (C,H,W)
        mask, g, b = orient
    elif orient.dim() == 3 and orient.size(2) == 3:        # (H,W,C)
        mask, g, b = orient[..., 0], orient[..., 1], orient[..., 2]
    else:
        return to_image_np(orient.mean(0, keepdim=True).expand(3, -1, -1))

    dx = g * 2.0 - 1.0
    dy = b * 2.0 - 1.0
    hue = (torch.atan2(dy, dx) + math.pi) / (2 * math.pi)
    sat = val = mask.clamp(0, 1)

    hsv = torch.stack([hue, sat, val], 0)
    rgb = matplotlib.colors.hsv_to_rgb(hsv.permute(1, 2, 0).cpu().numpy())
    return (rgb * 255).astype(np.uint8)

def rgbify(arr: np.ndarray) -> np.ndarray:
    return np.repeat(arr, 3, 2) if arr.ndim == 3 and arr.shape[2] == 1 else arr

# ─────────────────────────── main ─────────────────────────
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("perm-visualize")
    lp = ModelParams(parser); op = OptimizationParams(parser); pp = PipelineParams(parser)

    parser.add_argument("--start_checkpoint", type=str, required=True)
    parser.add_argument("--idname",           type=str, required=True)
    parser.add_argument("--image_res",        type=int, default=512)
    parser.add_argument("--out_dir",          type=str, required=True)
    parser.add_argument("--fps",              type=int, default=30)
    args = parser.parse_args(); args.device = "cuda"

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    lpt,opt,ppt = lp.extract(args), op.extract(args), pp.extract(args)

    scalp_mask = pickle.load(open(
        "/home/alrivero/ENPC/re_perm/flame/FLAME_masks/FLAME_masks.pkl","rb"),
        encoding="latin1")["scalp"]

    data_dir = Path(args.source_path)/args.idname
    mica_dir = data_dir/"track_out"/args.idname

    perm = Perm(lpt.perm_path, lpt.obj_head_path,
                scalp_vertex_idxs=scalp_mask,
                scalp_bounds=[0.1870,0.8018,0.4011,0.8047],
                mesh_scale=100.).to(args.device)
    pseudo = perm.hair_roots.load_txt(lpt.loaded_roots_path)[0]
    start_hair = np.load(lpt.emp_hair_path) if lpt.emp_hair_path else None
    gauss = GaussianPerm(perm, pseudo, start_hair, lpt.sh_degree).to(args.device)
    deform= PermDeformModel(perm, args.device).to(args.device)

    m,g,_ = torch.load(args.start_checkpoint, map_location=args.device)
    deform.restore(m); gauss.restore(g,opt); gauss.eval(); deform.eval()

    bg = torch.tensor([1,1,1] if lpt.white_background else [0,1,0],
                      dtype=torch.float32, device=args.device)

    scene = Scene_mica(str(data_dir), str(mica_dir),
                       train_type=0, white_background=lpt.white_background,
                       device=args.device)
    cams = scene.getCameras()

    vr, vs, vd, vcmp = [], [], [], []

    for cam in tqdm(cams, desc="rendering"):
        cam.load2device(args.device)

        cd = dict(R=torch.tensor(cam.R,device=args.device),
                  T=torch.tensor(cam.T,device=args.device),
                  roots=gauss.get_roots_xyz[None],
                  theta=gauss.theta, beta=gauss.beta)
        verts,_,rot_d,sc_c = deform.decode(gauss, cd)
        gauss.update_xyz_rot_scale(verts, rot_d, sc_c)

        pkg = render(cam, gauss, ppt, bg)

        pred_rgb = to_image_np(pkg["render"])
        pred_seg = to_image_np(pkg["segment"])
        pred_d   = depth_to_rgb(pkg["depth"][0], cam.hair_mask, invert=True)  # ← invert

        gt_full = to_image_np(cam.original_image)
        mask_np = (cam.hair_mask[0].cpu().numpy() > 0.5)
        gt_rgb  = gt_full * mask_np[..., None]
        gt_seg  = to_image_np(cam.hair_mask.float())
        gt_d    = depth_to_rgb(cam.depth_map, cam.hair_mask, invert=False)

        orient  = orient_to_rgb(cam.hair_orient) if hasattr(cam,"hair_orient") else gt_seg

        vr.append(pred_rgb); vs.append(pred_seg); vd.append(pred_d)

        strip = np.concatenate([
            rgbify(gt_rgb),  rgbify(pred_rgb),
            rgbify(gt_seg),  rgbify(pred_seg),
            rgbify(gt_d),    rgbify(pred_d),
            rgbify(orient)
        ], axis=1)
        vcmp.append(strip)

        cam.load2device("cpu")

    imageio.mimsave(Path(args.out_dir)/"render.mp4",       vr,   fps=args.fps)
    imageio.mimsave(Path(args.out_dir)/"segmentation.mp4", vs,   fps=args.fps)
    imageio.mimsave(Path(args.out_dir)/"depth.mp4",        vd,   fps=args.fps)
    imageio.mimsave(Path(args.out_dir)/"comparison.mp4",   vcmp, fps=args.fps)
    print(f"[✓] Videos saved in {args.out_dir}")

if __name__ == "__main__":
    main()