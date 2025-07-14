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
import debug
import trimesh
import cv2

from flame import FlameHead
from hair.hair_models          import Perm
from scene.gaussian_perm       import GaussianPerm
from scene                     import Scene_mica
from src.perm_deform_model     import PermDeformModel
from gaussian_renderer         import render
from arguments                 import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils       import to_image_np, compute_occlusion_mask
from scene.specular_model import SpecularModel
from scene.uncertainty_model import UncertaintyModel

STRAND_VERTEX_COUNT = 100

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
                      q_lo:  float = 0.6,
                      q_hi:  float = 1.0) -> torch.Tensor:
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
    """
    Collapse 360°→180°: treat θ and θ+180° as identical,
    and map that half-circle to the first 180° of the HSV wheel.
    """
    if orient.dim() == 3 and orient.size(0) == 3:          
        mask, g, b = orient
    elif orient.dim() == 3 and orient.size(2) == 3:        
        mask, g, b = orient[..., 0], orient[..., 1], orient[..., 2]
    else:
        return to_image_np(orient.mean(0, keepdim=True).expand(3, -1, -1))

    dx = g * 2.0 - 1.0
    dy = b * 2.0 - 1.0

    # raw angle in [0,2π)
    raw = torch.atan2(dy, dx) + math.pi

    # fold into [0,π)
    folded = torch.remainder(raw, math.pi)

    # map to [0,0.5] for 180° of hue
    hue = folded / (2 * math.pi)

    sat = val = mask.clamp(0, 1)

    hsv = torch.stack([hue, sat, val], 0)                 # (3,H,W)
    hsv_np = hsv.permute(1, 2, 0).cpu().numpy()            # (H,W,3)
    rgb   = matplotlib.colors.hsv_to_rgb(hsv_np)          # (H,W,3)
    return (rgb * 255).astype(np.uint8)

def orient_to_rgb_360(orient: torch.Tensor) -> np.ndarray:
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

import numpy as np
import cv2
from pathlib import Path
from typing  import List, Tuple

EPS = 1e-8  # numerical safety for divisions

# ───────────────────────────────── squeeze helper ────────────────────────────
def _squeeze(mask: np.ndarray) -> np.ndarray:
    """
    Accept (H,W), (H,W,1), or (1,H,W) and return (H,W) float32 in [0,1].
    """
    if mask.ndim == 3 and mask.shape[-1] == 1:        # (H,W,1)
        mask = mask[..., 0]
    if mask.ndim == 3 and mask.shape[0] == 1:         # (1,H,W)
        mask = mask[0]
    return mask.astype(np.float32)

# ───────────────────────────────── soft metrics ──────────────────────────────
def _soft_metrics(gt: np.ndarray,
                  pred: np.ndarray) -> dict:
    """
    Continuous TP / FP / FN statistics for probability masks.
    Returns precision, recall, dice, iou, fp_rate, fn_rate.
    """
    TP = np.sum(pred * gt)
    FP = np.sum(pred * (1.0 - gt))
    FN = np.sum((1.0 - pred) * gt)

    precision = TP / (TP + FP + EPS)
    recall    = TP / (TP + FN + EPS)
    dice      = 2 * TP / (2 * TP + FP + FN + EPS)
    iou       = TP / (TP + FP + FN + EPS)
    fp_rate   = FP / (TP + FN + EPS)   # FP relative to GT positives
    fn_rate   = FN / (TP + FN + EPS)

    return dict(precision=precision, recall=recall,
                dice=dice, iou=iou,
                fp_rate=fp_rate, fn_rate=fn_rate)

# ─────────────────────────── visualisation + metrics ─────────────────────────
def _visualise_and_metrics(gt: np.ndarray,
                           pred: np.ndarray,
                           vis_thresh: float = 0.0) -> Tuple[np.ndarray, dict]:
    """
    * Metrics are soft/continuous.
    * Visualisation bins both masks at vis_thresh only for colouring.
    """
    G = _squeeze(gt)
    P = _squeeze(pred)

    metrics = _soft_metrics(G, P)           # ← continuous!

    # ----- make colour overlay for display -----
    G_bin = G > vis_thresh
    P_bin = P > vis_thresh

    H, W = G_bin.shape
    vis  = np.zeros((H, W, 3), dtype=np.uint8)
    vis[np.logical_and(G_bin,  P_bin)] = (255, 255, 255)  # TP  (white)
    vis[np.logical_and(G_bin, ~P_bin)] = (255,   0,   0)  # FN  (red)
    vis[np.logical_and(~G_bin, P_bin)] = (  0,   0, 255)  # FP  (blue)

    return vis, metrics

# ───────────────────────── video + metrics writer ────────────────────────────
def save_mask_comparison_video(pred_list: List[np.ndarray],
                               gt_list:   List[np.ndarray],
                               out_dir:   str,
                               out_name:  str = "mask_comparison.mp4",
                               fps:       int = 24,
                               vis_thresh: float = 0.0):
    """
    Creates a side-by-side video  [ GT | Diff-visualisation ]
    and saves per-frame continuous-metrics to metrics.npy.
    """
    assert len(pred_list) == len(gt_list), "Pred / GT list length mismatch"

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ——— setup video writer ———
    first_vis, _ = _visualise_and_metrics(gt_list[0], pred_list[0], vis_thresh)
    H, W, _ = first_vis.shape
    canvas_size = (W * 2, H)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path / out_name),
                             fourcc, fps, canvas_size)

    metrics_all = []
    for idx, (pred, gt) in enumerate(zip(pred_list, gt_list)):
        vis, m = _visualise_and_metrics(gt, pred, vis_thresh)
        metrics_all.append(m)

        # assemble side-by-side frame
        gt_vis = (_squeeze(gt) * 255).astype(np.uint8)
        gt_vis = np.repeat(gt_vis[..., None], 3, axis=2)
        frame  = np.hstack([gt_vis, vis])            # (H,2W,3)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        print(f"Frame {idx:04d}:  "
              f"IoU={m['iou']:.3f}  Dice={m['dice']:.3f}  "
              f"Prec={m['precision']:.3f}  Rec={m['recall']:.3f}")

    writer.release()

    # ——— save metrics.npy ———
    dtype = [('precision','f4'),('recall','f4'),
             ('dice','f4'),('iou','f4'),
             ('fp_rate','f4'),('fn_rate','f4')]
    mets_np = np.array([tuple(d.values()) for d in metrics_all], dtype=dtype)
    np.save(out_path / "metrics.npy", mets_np)

    # ——— print dataset means ———
    mean_vals = {k: float(mets_np[k].mean()) for k in mets_np.dtype.names}
    print("\n=== Mean over all frames ===")
    for k, v in mean_vals.items():
        print(f"{k:10s}: {v:.4f}")

    print(f"\nVideo saved to {out_path/out_name}")
    print(f"Per-frame metrics saved to {out_path/'metrics.npy'}")

# ─────────────────────────── main ─────────────────────────
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("perm-visualize")
    lp = ModelParams(parser); op = OptimizationParams(parser); pp = PipelineParams(parser)

    parser.add_argument("--start_checkpoint", type=str, required=True)
    parser.add_argument("--idname",           type=str, required=True)
    parser.add_argument("--image_res",        type=int, default=720)
    parser.add_argument("--out_dir",          type=str, required=True)
    parser.add_argument("--fps",              type=int, default=30)
    parser.add_argument("--cull_head_off",              type=bool, default=True)
    args = parser.parse_args(); args.device = "cuda"

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    lpt,opt,ppt = lp.extract(args), op.extract(args), pp.extract(args)

    data_dir     = os.path.join(args.source_path, args.idname)
    log_dir      = os.path.join(data_dir, "log")
    model_dir    = os.path.join(log_dir, "ckpt")

    scalp_mask = pickle.load(open(
        "/home/alrivero/ENPC/re_perm/flame/assets/FLAME_masks.pkl","rb"),
        encoding="latin1")["scalp"]

    data_dir = Path(args.source_path)/args.idname
    mica_dir = data_dir/"track_out"/args.idname

    flame = FlameHead(
        300, 
        100, 
        add_teeth=True,
        remove_lip_inside=False,
        face_clusters=("skin", "hair", "boundary", "lips_tight", "teeth", "sclerae", "irises"),
    ).to(args.device)

    perm = Perm(lpt.perm_path, lpt.obj_head_path,
                scalp_vertex_idxs=scalp_mask,
                scalp_bounds=[0.1870,0.8018,0.4011,0.8047],
                mesh_scale=100.).to(args.device)
    pseudo = perm.hair_roots.load_txt(lpt.loaded_roots_path)[0]
    start_hair = np.load(lpt.emp_hair_path) if lpt.emp_hair_path else None
    gauss = GaussianPerm(perm, pseudo, start_hair, lpt.sh_degree, lpt.asg_degree).to(args.device)
    deform= PermDeformModel(perm, flame, args.device).to(args.device)

    rotation_offsets = torch.load("/data/add_disk0/alrivero/imagine_data/arjit/log/ckpt/flame_rot_060000.pth")
    translation_offsets = torch.load("/data/add_disk0/alrivero/imagine_data/arjit/log/ckpt/flame_trans_060000.pth")

    extra_parameters = [
        {"params": [rotation_offsets], "lr": 0.0001, "name": "flame_rot"},
        {"params": [translation_offsets], "lr": 0.0001, "name": "flame_trans"}
    ]
    gauss.training_setup(opt, extra_parameters=extra_parameters)

    specular_mlp = SpecularModel()
    specular_mlp.specular = specular_mlp.specular.to(args.device)
    specular_mlp.train_setting(opt)

    uncertainty_mlp = UncertaintyModel()
    uncertainty_mlp.uncertainty = uncertainty_mlp.uncertainty.to(args.device)
    uncertainty_mlp.train_setting(opt)

    m,g,it = torch.load(args.start_checkpoint, map_location=args.device)
    deform.restore(m); gauss.restore(g,opt, extra_parameters=extra_parameters); gauss.eval(); deform.eval()
    specular_mlp.load_weights(model_dir, iteration=it)
    uncertainty_mlp.load_weights(model_dir, iteration=it)

    bg = torch.tensor([1,1,1] if lpt.white_background else [0,1,0],
                      dtype=torch.float32, device=args.device)

    scene = Scene_mica(str(data_dir), str(mica_dir),
                       train_type=0, white_background=lpt.white_background,
                       device=args.device)
    cams = scene.getCameras()

    vr, vs, vd, vcmp = [], [], [], []

    all_pred_masks = []
    all_gt_masks = []

    load_filter = True
    for cam in tqdm(cams, desc="rendering"):
        cam.load2device(args.device)

        enable_flame_offsets = lpt.learn_flame_rigid_offset and it >= 10000
        flame_rot = rotation_offsets[cam.uid][None] if enable_flame_offsets else torch.tensor([[0.0, 0.0, 0.0]]).to(args.device)
        flame_trans = translation_offsets[cam.uid][None] if enable_flame_offsets else torch.tensor([[0.0, 0.0, 0.0]]).to(args.device)
        
        codedict = {
            "R":      torch.tensor(cam.R, device=args.device),
            "T":      torch.tensor(cam.T, device=args.device),
            "roots":  gauss.get_roots_xyz[None],
            "theta":  gauss.theta,
            "beta":   gauss.beta,
            "expr":   cam.exp_param,
            "shape":   cam.shape_param,
            "eyes_pose":   cam.eyes_pose,
            "jaw_pose":   cam.jaw_pose,
            "neck_pose":   cam.neck_pose,
            "root_pose":   torch.zeros_like(flame_rot),
            "translation":   torch.zeros_like(flame_trans)
        }

        flame_verts, flame_faces, _ = flame(
            codedict['shape'],
            codedict['expr'],
            torch.zeros_like(flame_rot),
            codedict['neck_pose'],
            codedict['jaw_pose'],
            codedict['eyes_pose'],
            torch.zeros_like(flame_trans),
            return_faces=True,
            return_normals=True,
        )
        
        _, _, verts, _, rot_d, sc_c = deform.decode(gauss, codedict)
        strand_pts = verts.reshape(gauss.num_strands, STRAND_VERTEX_COUNT, 3)
        tangents = gauss.update_xyz_rot_scale(strand_pts, rot_d, sc_c)

        if load_filter:
            gauss.compute_3D_filter(cams, args.device)
            # gauss.prune_strands_by_opacity(0.01)
            # gauss.compute_3D_filter(cams, args.device)
            load_filter = False

        if args.cull_head_off:
            occ_mask, depth_map = compute_occlusion_mask(gauss, cam, flame_verts, flame_faces)
        else:
            occ_mask, depth_map = None, None

        dir_pp = (gauss.get_xyz - cam.camera_center.repeat(gauss.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        spec_color = specular_mlp.step(gauss.get_asg_features, dir_pp_normalized, tangents)

        dir_pp = (gauss.get_xyz - cam.camera_center.repeat(gauss.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        uncertainty_vals = uncertainty_mlp.step(
            gauss.get_gate_per_gaussian,
            dir_pp_normalized,
            tangents,
            gauss.get_axial_weight,
            torch.tensor(cam.uid / 1879).to(args.device)
        )

        pkg = render(
            cam,
            gauss,
            ppt,
            bg,
            kernel_size=lpt.kernel_size,
            occ_mask=occ_mask,
            spec_color=spec_color,
            uncertainty_vals=uncertainty_vals
        )

        pred_rgb = to_image_np(pkg["render"])
        pred_seg = to_image_np(pkg["gate"])

        gt_full = to_image_np(cam.original_image)
        mask_np = (cam.hair_mask[0].cpu().numpy() > 0.5)
        gt_rgb  = gt_full * mask_np[..., None]
        gt_seg  = to_image_np(cam.hair_mask.float())

        orient  = orient_to_rgb(cam.hair_orient) if hasattr(cam,"hair_orient") else gt_seg
        orient = cv2.resize(orient, (720, 720), interpolation=cv2.INTER_LINEAR)

        all_pred_masks.append(pred_seg)
        all_gt_masks.append(gt_seg)

        vr.append(pred_rgb); vs.append(pred_seg)

        strip = np.concatenate([
            rgbify(gt_rgb),  rgbify(pred_rgb),
            rgbify(gt_seg),  rgbify(pred_seg),
            rgbify(orient),
        ], axis=1)
        if depth_map is not None:
            strip = np.concatenate([strip, rgbify(to_image_np(depth_map))], axis=1)

        vcmp.append(strip)

        cam.load2device("cpu")

    import pdb; pdb.set_trace()
    save_mask_comparison_video(all_pred_masks, all_gt_masks, Path(args.out_dir), fps=args.fps)

    imageio.mimsave(Path(args.out_dir)/"render.mp4",       vr,   fps=args.fps)
    imageio.mimsave(Path(args.out_dir)/"segmentation.mp4", vs,   fps=args.fps)
    # imageio.mimsave(Path(args.out_dir)/"depth.mp4",        vd,   fps=args.fps)
    imageio.mimsave(Path(args.out_dir)/"comparison.mp4",   vcmp, fps=args.fps)
    print(f"[✓] Videos saved in {args.out_dir}")

if __name__ == "__main__":
    main()