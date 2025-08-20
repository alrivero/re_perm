import os, sys, random, argparse, pickle, cv2
import numpy as np
import torch
import torch.nn as nn
import lpips
import datetime as dt
import debug
import yaml
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pathlib
import trimesh
from typing import Union, Optional

try:
    import wandb
    _use_wandb = False
except ModuleNotFoundError:
    print("[wandb] not found – continuing without online logging.")
    _use_wandb = False

# --------------------------------------------------------------------- #
# project imports
# --------------------------------------------------------------------- #
from hair.hair_models import Perm
from scene.gaussian_perm import GaussianPerm
from scene.specular_model import SpecularModel
from scene.uncertainty_model import UncertaintyModel
from scene import Scene_mica
from src.perm_deform_model import PermDeformModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import save_tensor_to_ply, export_strands_as_obj, export_strands_to_usd, save_tensor_to_obj, average_opacity_for_strand, average_opacity_per_strand, compute_occlusion_mask, create_outside_mask
from utils.loss_utils import (
    orientation_loss_v2_debug,
    neighbour_orientation_loss,
    bending_loss,
    head_collision_loss,
    gaussian_head_collision_loss,
    local_length_consistency_loss,
    color_variance_loss_sh,
    opacity_variance_loss,
    asg_variance_loss,
    gaussian_scale_regularization_loss,
    triangle_scale_area_loss,
    HairDetailLoss,
    MaskShapeKLWithExtras, 
    alpha_blended_loss,
    geometric_fit_loss
)

STRAND_VERTEX_COUNT = 100    # same as in GaussianPerm


# --------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------- #
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_image_np(tensor: torch.Tensor) -> np.ndarray:
    """
    This function is a general utility and does not need to be changed.
    It correctly handles tensors of shape (C, H, W) or (H, W).
    """
    t = tensor.detach().clamp(0, 1)
    if t.dim() == 4:
        t = t[0]
    if t.dim() == 3:
        arr = t.permute(1, 2, 0).cpu().numpy()
    elif t.dim() == 2:
        gray = t.cpu().numpy()
        arr = np.stack([gray, gray, gray], axis=-1)
    else:
        raise RuntimeError(f"Cannot convert tensor shape {tuple(t.shape)} to image")
    return (arr * 255.0).astype(np.uint8)


def make_side_by_side(left: torch.Tensor, right: torch.Tensor, image_res: tuple) -> np.ndarray:
    """
    Creates a side-by-side comparison image from two tensors.
    
    Args:
        image_res (tuple): The image resolution as (WIDTH, HEIGHT).
    """
    left_np  = to_image_np(left)
    right_np = to_image_np(right)
    
    # --- FIX: Unpack width and height from the tuple ---
    # Note: argparse gives (W, H), but NumPy/OpenCV use (H, W) for shape
    height, width = image_res
    
    # Create a canvas that is twice the width
    canvas = np.zeros((height, width * 2, 3), dtype=np.uint8)
    
    # Place the images correctly using the actual width
    canvas[:, :width] = left_np
    canvas[:, width:] = right_np
    return canvas


def save_gate_vis(
    gate_tensor: torch.Tensor,
    out_path: Union[str, pathlib.Path],
    *,
    alpha_mask: Optional[torch.Tensor] = None,
    cmap: str = "plasma",
    dpi: int = 150,
):
    """
    Visualize the gate map values, preserving the aspect ratio.
    """
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gate = gate_tensor.squeeze().float().detach().cpu()
    if alpha_mask is not None:
        mask = alpha_mask.squeeze().bool().cpu()
    else:
        mask = torch.ones_like(gate, dtype=torch.bool)

    gate_vis = torch.zeros_like(gate)
    gate_vis[mask] = gate[mask]

    # --- FIX: Calculate figsize dynamically to preserve aspect ratio ---
    height, width = gate_vis.shape
    base_size = 4.0  # Inches for the smaller dimension
    if width >= height:
        figsize = (base_size * (width / height), base_size)
    else:
        figsize = (base_size, base_size * (height / width))
    # --- END FIX ---

    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(gate_vis.numpy(), cmap=cmap, vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.title(out_path.name, fontsize=10)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()

def toggle_beta_trainable(gaussians, trainable: bool, training_args):
    """
    During warm-up: freeze beta, color, and scale parameters.
    After warm-up: rebuild optimizer and enable them again.
    """
    if not trainable:
        # zero out beta, color, and scale learning‐rates
        for pg in gaussians.optimizer.param_groups:
            if pg.get("name") in ("beta", "f_dc", "f_rest"):
                pg["lr"] = 0.0

        # disable gradients
        gaussians.beta.requires_grad_(False)
        gaussians._features_dc.requires_grad_(False)
        gaussians._features_rest.requires_grad_(False)
    else:
        # rebuild optimizer & LR schedules from scratch
        gaussians.training_setup(training_args)
        
        # re-enable gradients
        gaussians.beta.requires_grad_(True)
        gaussians._features_dc.requires_grad_(True)
        gaussians._features_rest.requires_grad_(True)


# --------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    # 1. General argument setup
    parser = argparse.ArgumentParser("perm-fitting")

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--idname", type=str, default="id1_25")
    parser.add_argument('--image_res', type=int, nargs=2, default=[1280, 720], metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument("--start_checkpoint", type=str, default=None)
    # Ensure OptimizationParams defines:
    #   lambda_color_var, lambda_local_len, lambda_strand_rep
    #   k_strand_rep, step_strand_rep, safe_dist_strand_rep
    # e.g. parser.add_argument("--lambda_strand_rep", type=float, default=0.0)

    args = parser.parse_args(sys.argv[1:])
    args.device = "cuda"

    set_random_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)

    lpt = lp.extract(args)
    opt = op.extract(args)
    ppt = pp.extract(args)

    if _use_wandb:
        run_name = f"{args.idname}—{dt.datetime.now().strftime('%b-%d-%Y_%Hh%Mm')}"
        wandb.init(
            project="re_perm",
            name=run_name,
            config={**args.__dict__, **opt.__dict__, **ppt.__dict__, **lpt.__dict__}
        )

    data_dir     = os.path.join(args.source_path, args.idname)
    log_dir      = os.path.join(data_dir, "log")
    train_dir    = os.path.join(log_dir, "train")
    model_dir    = os.path.join(log_dir, "ckpt")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 2. Load in Scene
    scene = Scene_mica(
        data_dir,
        white_background=lpt.white_background,
        device=args.device,
        img_dim=args.image_res
    )
    all_cameras = scene.getCameras().copy()
    viewpoint_stack = None


    # 3. SMPL-X/FLAME Set-Up
    head = trimesh.load(lpt.obj_head_path)
    og_pos_trans = head.vertices.mean(axis=0)
    smplx_joints = torch.load(args.joints_smplx)
    smplx_params = np.load(args.data_smplx, allow_pickle=True)
    valid_scalp_idxs = pickle.load(open(args.vertex_idxs_scalp, "rb"))

    # 4. PERM Set-Up
    perm = Perm(
        lpt.perm_path,
        lpt.obj_head_path,
        scalp_vertex_idxs=valid_scalp_idxs,
        scalp_bounds=[0.1870, 0.8018, 0.4011, 0.8047],
        mesh_scale=100.0,
        mesh_translate=[-5.5688e-05,  7.5134e-05,  1.0533e-04],
        mean_center_before_rigid=True
    ).to(args.device)
    perm = perm.eval()

    # 5. Gaussian PERM Scene Set-Up
    start_hair_style = None
    if lpt.emp_hair_path:
        start_hair_style = np.load(lpt.emp_hair_path)

    gaussians = GaussianPerm(perm, start_hair_style, lpt.sh_degree, lpt.asg_degree, smplx_params["global_scale"].item()).to(args.device)
    gaussians.roots = gaussians.roots.to(args.device)
    save_tensor_to_obj(gaussians.roots, os.path.join(train_dir, "roots.obj"))

    # 6. Optional Trainable Rot/Trans Residual
    extra_parameters = None
    if lpt.learn_flame_rigid_offset:
        rotation_offsets = torch.eye(3,
                            dtype=torch.float32,
                            device=args.device)
        rotation_offsets = rotation_offsets.unsqueeze(0).repeat(len(scene.getCameras()), 1, 1).requires_grad_()
        translation_offsets = torch.tensor([0.0, 0.0, 0.0],
                                dtype=torch.float32,
                                device=args.device)
        translation_offsets = translation_offsets.unsqueeze(0).repeat(len(scene.getCameras()), 1).requires_grad_()

        # ADD ME IN
        # self.cam_rotation_lr = 0.001
        # self.cam_translation_lr_init = 0.0016
        # self.cam_translation_lr_final = 0.000016

        extra_parameters = [
            {"params": [rotation_offsets], "lr": 0.0001, "name": "flame_rot"},
            {"params": [translation_offsets], "lr": 0.0001, "name": "flame_trans"}
        ]

        torch.save(rotation_offsets, os.path.join(model_dir, f"flame_rot.pth"))

    # 7. Training Set-Up
    gaussians.training_setup(opt, extra_parameters=extra_parameters)

    # 8. Initially Freeze Beta (Not really used)
    # toggle_beta_trainable(gaussians, False, opt)

    # 9. Define Deform/Specular/Uncertainty Models
    deform_model = PermDeformModel(
        perm,
        og_pos_trans,
        smplx_joints,
        smplx_params,
        args.device
    ).to(args.device)

    specular_mlp = SpecularModel()
    specular_mlp.specular = specular_mlp.specular.to(args.device)
    specular_mlp.train_setting(opt)

    uncertainty_mlp = UncertaintyModel()
    uncertainty_mlp.uncertainty = uncertainty_mlp.uncertainty.to(args.device)
    uncertainty_mlp.train_setting(opt)

    first_iter = 0
    uniform_strand_color = False
    enable_uncertainty = False
    densify_count = 1
    densify_on_start = False

    if args.start_checkpoint:
        (g_params, first_iter, uniform_strand_color, enable_uncertainty, densify_count, densify_on_start, opt.densification_strand_interval, opt.densify_from_iter, opt.lambda_color_var, opt.lambda_opacity_var) = torch.load(args.start_checkpoint)
        gaussians.restore(g_params, opt, extra_parameters=extra_parameters)
        specular_mlp.load_weights(model_dir, iteration=first_iter)
        uncertainty_mlp.load_weights(model_dir, iteration=first_iter)
        first_iter -= 1

        # if lpt.learn_flame_rigid_offset:
        #     rotation_offsets = torch.load("/data/add_disk0/alrivero/imagine_data/arjit/log/ckpt/flame_rot_070000.pth")
        #     translation_offsets = torch.load("/data/add_disk0/alrivero/imagine_data/arjit/log/ckpt/flame_trans_070000.pth")

    bg_color = [1, 1, 1] if lpt.white_background else [0, 1, 0]
    bg_image = torch.zeros((3, args.image_res[0], args.image_res[1]), device=args.device)
    if lpt.white_background:
        bg_image[:] = 1
    else:
        bg_image[1] = 1
    background = torch.tensor(bg_color, dtype=torch.float32, device=args.device)

    hair_photo_loss = HairDetailLoss(
        warmup_iters=8000,
        fade_iters=12000,
        polish_iters=5000,
        blur=True,
        device=args.device
    )

    kl_loss = MaskShapeKLWithExtras(
        pi_star = scene.target_dist['pi'],
        alpha_star = scene.target_dist['alpha'],
        beta_star  = scene.target_dist['beta'],
        target_nz_frac= scene.target_dist['non_one_frac'],
        spike_thresh = 254/255,       # same as during fitting
        beta_start = 1e-3,
        beta_final = 5e-2,
        t_start    = 8000,
        t_end      = 50_000,
        warmup_iter=8_000
    )


    head_faces = torch.tensor(list(perm.hair_roots.head.faces)).to(args.device)
    bp = False

    for it in range(first_iter + 1, opt.iterations + 1):
        if it % 500 == 0:
            gaussians.oneupSHdegree()

        # if it == opt.theta_warmup + 1:
        #     toggle_beta_trainable(gaussians, True, opt)

        # if it % 3000 == 0 and it != first_iter + 1:
        #     sample_pow += 1
        #     sample_idxs = list(range(0, len(all_cameras), int(len(all_cameras) / min(len(all_cameras), (4 ** sample_pow)))))

        if not viewpoint_stack:
            viewpoint_stack = scene.getCameras().copy().tolist()
            random.shuffle(viewpoint_stack)
        cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack) - 1))
        cam.load2device(args.device)

        enable_flame_offsets = lpt.learn_flame_rigid_offset and it >= 10000
        flame_rot = rotation_offsets[cam.colmap_id] if enable_flame_offsets else torch.eye(3).to(args.device)
        flame_trans = translation_offsets[cam.colmap_id] if enable_flame_offsets else torch.tensor([[0.0, 0.0, 0.0]]).to(args.device)
        
        codedict = {
            "R":      torch.tensor(cam.R, device=args.device),
            "T":      torch.tensor(cam.T, device=args.device),
            "roots":  gaussians.get_roots_xyz[None],
            "theta":  gaussians.theta,
            "beta":   gaussians.beta,
            "learned_rot":   flame_rot,
            "learned_trans":   flame_trans
        }

        (
           verts_final,
           guide_final,
           verts_final_def,
           guide_final_def,
           scalp_final,
           scalp_final_def,
           head_final,
           head_final_def,
           normals_final,
           normals_final_def
        ) = deform_model.decode(gaussians, codedict)
        strand_pts_can = verts_final.reshape(gaussians.num_strands, STRAND_VERTEX_COUNT, 3)
        strand_pts = verts_final_def.reshape(gaussians.num_strands, STRAND_VERTEX_COUNT, 3)

        if bp:
            import pdb; pdb.set_trace()
        tangents = gaussians.update_xyz_rot_scale(strand_pts, scalp_final_def)

        gt_img    = cam.original_image
        gt_alpha  = cam.hair_mask
        occ_mask, depth_map = compute_occlusion_mask(gaussians, cam, head_final_def[None], head_faces[None])
        outside_mask = create_outside_mask(gaussians, cam, occ_mask, gt_alpha)

        if it == first_iter + 1:
            gaussians.set_scalp_opacity(scalp_final)
            gaussians.compute_3D_filter(cameras=all_cameras, device=args.device)

        if it >= 3000:
            dir_pp = (gaussians.get_xyz - cam.camera_center.repeat(gaussians.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            spec_color = specular_mlp.step(gaussians.get_asg_features, dir_pp_normalized, tangents)
        else:
            spec_color = 0.0

        if enable_uncertainty:
            dir_pp = (gaussians.get_xyz - cam.camera_center.repeat(gaussians.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            is_scalp = torch.ones(dir_pp.shape[0]).to(args.device)
            is_scalp[-gaussians.num_strands:] = 0.0

            uncertainty_vals = uncertainty_mlp.step(
                gaussians.get_gate_per_gaussian,
                dir_pp_normalized,
                tangents,
                gaussians.get_axial_weight,
                is_scalp
            )
        else:
            uncertainty_vals = None


        render_pkg = render(cam, gaussians, ppt, background, kernel_size=lpt.kernel_size, occ_mask=occ_mask, spec_color=spec_color, uncertainty_vals=uncertainty_vals)
        img_render   = render_pkg["render"]
        img_segment  = render_pkg["segment"]

        alpha = render_pkg["gate"]

        if enable_uncertainty:
            # import pdb; pdb.set_trace()
            loss_uncertainty_kl, loss_dice, loss_nz_frac, kappa = kl_loss(alpha, gt_alpha, it)
        else:
            loss_uncertainty_kl = torch.tensor(0.0).to(args.device)
            loss_dice = torch.tensor(0.0).to(args.device)
            loss_nz_frac = torch.tensor(0.0).to(args.device)
            kappa = 1.0

        orient    = cam.hair_orient
        
        if enable_uncertainty:
            loss_h = kappa * hair_photo_loss(img_render, gt_img, it, gate_map=alpha.detach())
            loss_seg = kappa * alpha_blended_loss(alpha, gt_alpha)
        else:
            gt_img    = gt_img * gt_alpha + bg_image * (1 - gt_alpha)
            loss_h = hair_photo_loss(img_render, gt_img, it)
            loss_seg = alpha_blended_loss(img_segment, gt_alpha)

        guide_pts = guide_final.reshape(-1, STRAND_VERTEX_COUNT, 3)

        loss_o                    = kappa * orientation_loss_v2_debug(cam, gaussians, alpha, orient, occ_mask)
        loss_geom_fit             = geometric_fit_loss(gaussians)
        loss_nei                  = neighbour_orientation_loss(strand_pts[:, :STRAND_VERTEX_COUNT, :], gaussians.neighbor_idx)
        loss_bend                 = bending_loss(strand_pts)
        loss_sobel                = torch.tensor(0.0).to(args.device)
        loss_head_col             = head_collision_loss(strand_pts, head_final_def, normals_final_def)
        loss_gauss_head_col       = gaussian_head_collision_loss(gaussians, head_final_def, normals_final_def)
        loss_local_len            = local_length_consistency_loss(gaussians)
        loss_scale_reg            = gaussian_scale_regularization_loss(gaussians)
        
        if uniform_strand_color:
            loss_color_var = torch.tensor(0.0).to(args.device)
            loss_asg_var = torch.tensor(0.0).to(args.device)
            loss_opacity_var = torch.tensor(0.0).to(args.device)
        else:
            loss_color_var            = color_variance_loss_sh(gaussians)
            loss_asg_var              = asg_variance_loss(gaussians)
            loss_opacity_var          = opacity_variance_loss(gaussians)

        if enable_flame_offsets:
            loss_flame_rot_reg = (rotation_offsets.norm(dim=1) ** 2).mean()
            loss_flame_trans_reg = (translation_offsets.norm(dim=1) ** 2).mean()
        else:
            # keep the graph happy when offsets are frozen / disabled
            loss_flame_rot_reg   = torch.tensor(0.0, device=args.device)
            loss_flame_trans_reg = torch.tensor(0.0, device=args.device)


        mu_theta    = gaussians.theta_start.clone().to(args.device)
        mu_beta     = gaussians.beta_start.clone().to(args.device)
        loss_theta_l2 = (gaussians.theta - mu_theta).pow(2).sum()
        loss_beta_l2  = (gaussians.beta - mu_beta).pow(2).sum()

        lambda_huber          = opt.lambda_huber
        lambda_seg            = opt.lambda_seg
        lambda_sobel          = opt.lambda_sobel
        lambda_nei            = opt.lambda_neigh
        lambda_orient         = opt.lambda_orient
        lambda_geom_fit       = opt.lambda_geom_fit
        lambda_bend           = opt.lambda_bend
        lambda_head_col       = opt.lambda_head_col
        lambda_gauss_head_col = opt.lambda_gauss_head_col
        lambda_local_len      = opt.lambda_local_len
        lambda_color_var      = opt.lambda_color_var
        lambda_opacity_var    = opt.lambda_opacity_var
        lambda_asg_var        = opt.lambda_asg_var
        lambda_theta_l2       = opt.lambda_theta_l2
        lambda_beta_l2        = opt.lambda_beta_l2
        lambda_scale_reg      = opt.lambda_scale_reg
        lambda_flame_rot_reg   = opt.lambda_flame_rot_reg
        lambda_flame_trans_reg = opt.lambda_flame_trans_reg
        lambda_uncertainty_kl  = opt.lambda_uncertainty_kl
        lambda_dice            = opt.lambda_dice
        lambda_nz_frac         = opt.lambda_nz_frac

        w_huber           = lambda_huber                * loss_h.item()
        w_nei             = lambda_nei                  * loss_nei.item()
        w_orient          = lambda_orient               * loss_o.item()
        w_seg             = lambda_seg                  * loss_seg.item()
        w_geom_fit        = lambda_geom_fit             * loss_geom_fit.item()
        w_bend            = lambda_bend                 * loss_bend.item()
        w_sobel           = lambda_sobel                * loss_sobel.item()
        w_head_col        = lambda_head_col             * loss_head_col.item()
        w_gauss_head_col  = lambda_gauss_head_col       * loss_gauss_head_col.item()
        w_local_len       = lambda_local_len            * loss_local_len.item()
        w_color_var       = lambda_color_var            * loss_color_var.item()
        w_opacity_var     = lambda_opacity_var          * loss_opacity_var.item()
        w_asg_var         = lambda_asg_var              * loss_asg_var.item()
        w_theta_l2        = lambda_theta_l2             * loss_theta_l2.item()
        w_beta_l2         = lambda_beta_l2              * loss_beta_l2.item()
        w_scale_reg       = lambda_scale_reg            * loss_scale_reg.item()
        w_flame_rot_reg   = lambda_flame_rot_reg        * loss_flame_rot_reg.item()
        w_flame_trans_reg = lambda_flame_trans_reg      * loss_flame_trans_reg.item()
        w_uncertainty_kl  = lambda_uncertainty_kl       * loss_uncertainty_kl.item()
        w_dice            = lambda_dice                 * loss_dice.item()
        w_nz_frac         = lambda_nz_frac              * loss_nz_frac.item()

        loss = (
            lambda_huber           * loss_h +
            lambda_seg             * loss_seg +
            lambda_orient          * loss_o +
            lambda_geom_fit        * loss_geom_fit +
            lambda_nei             * loss_nei +
            lambda_bend            * loss_bend +
            lambda_sobel           * loss_sobel +
            lambda_head_col        * loss_head_col +
            lambda_gauss_head_col  * loss_gauss_head_col +
            lambda_local_len       * loss_local_len +
            lambda_color_var       * loss_color_var +
            lambda_opacity_var     * loss_opacity_var +
            lambda_asg_var         * loss_asg_var +
            lambda_theta_l2        * loss_theta_l2 +
            lambda_beta_l2         * loss_beta_l2 +
            lambda_scale_reg       * loss_scale_reg +
            lambda_flame_rot_reg   * loss_flame_rot_reg +
            lambda_flame_trans_reg * loss_flame_trans_reg +
            lambda_uncertainty_kl  * loss_uncertainty_kl +
            lambda_dice            * loss_dice +
            lambda_nz_frac         * loss_nz_frac
        )

        gaussians.update_learning_rate(it)
        loss.backward()

        # Densification
        radii                  = render_pkg["radii"]                 # (M,) for renderer
        visibility_filter      = render_pkg["visibility_filter"]     # (M,) for renderer
        viewspace_point_tensor = render_pkg["viewspace_points"]

        # track running ∥∇xy∥ statistics
        if bp:
            import pdb; pdb.set_trace()
        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, occ_mask, outside_mask)

        # densify / prune within schedule window
        # import pdb; pdb.set_trace()

        if densify_on_start or (it < opt.densify_strands_until_iter and it > opt.densify_strands_from_iter and it % opt.densification_strand_interval == 0):
            
            torch.save(
                (gaussians.capture(), it, uniform_strand_color, enable_uncertainty, densify_count, densify_on_start, opt.densification_strand_interval, opt.densify_from_iter, opt.lambda_color_var, opt.lambda_opacity_var),
                os.path.join(model_dir, f"chkpnt_{it:06d}_pre_dense.pth")
            )
            specular_mlp.save_weights(model_dir, it)
            uncertainty_mlp.save_weights(model_dir, it)
            print(f"\n[ITER {it}] Pre-Densification Checkpoint saved.\n")
            
            if not densify_on_start:
                densify_count += 1
                uniform_strand_color = densify_count >= 3

            new_roots, new_radii, new_roots_culled = perm.hair_roots.densify_scalp_hex()
            while new_roots_culled.shape[0] <= gaussians.roots.shape[0]:
                new_roots, new_radii, new_roots_culled = perm.hair_roots.densify_scalp_hex()

            gaussians.reset_gaussians_to_new_roots(new_roots_culled, new_radii, uniform_strand_color=uniform_strand_color, random_color=densify_on_start)
            gaussians.compute_3D_filter(cameras=all_cameras, device=args.device)

            gaussians.training_setup(opt, extra_parameters=extra_parameters)
            save_tensor_to_obj(gaussians.roots, os.path.join(train_dir, f"roots_{it}.obj"))

            if not densify_on_start:
                opt.densification_strand_interval *= 3
                opt.densify_from_iter = 1 + it
                opt.lambda_color_var *= 10.0
                opt.lambda_opacity_var *= 10.0
            densify_on_start = False
            gaussians.set_scalp_opacity(scalp_final)

        if it < opt.densify_until_iter and it > opt.densify_from_iter \
                                        and it % opt.densification_interval == 0:
            if occ_mask is not None:
                with torch.no_grad():
                    radii = render(cam, gaussians, ppt, background, kernel_size=lpt.kernel_size, spec_color=spec_color)["radii"]
                                        
            size_threshold = 20 if it > opt.opacity_reset_interval else None
            n_split, n_clone, n_merge, n_prune = gaussians.densify_and_prune(
                radii=radii,
                grad_thresh=opt.densify_grad_threshold,
                min_opac=opt.min_opacity 
            )
            print(f"\n[ITER {it}] Gaussians Cloned: {n_clone} Gaussians Split {n_split} Gaussians Merged {n_merge} Gaussians Pruned {n_prune}\n")
            print(f"\n[ITER {it}] Scale Min: {gaussians.get_scaling[:, 1].min()}, Scale Max: {gaussians.get_scaling[:, 1].max()}, Scale Median: {gaussians.get_scaling[:, 1].median()}\n")
            print(f"\n[ITER {it}] 5th Quantile: {gaussians.get_scaling[:, 1].quantile(0.05)}, 10th Quantile: {gaussians.get_scaling[:, 1].quantile(0.10)}, 15th Quantile: {gaussians.get_scaling[:, 1].quantile(0.15)}\n")

            num_reset = gaussians.halve_large_parallel_sigmas()
            print(f"Gaussians Scales Halved: {num_reset}")

            if n_clone + n_split + n_merge + n_prune > 0:
                gaussians.compute_3D_filter(cameras=all_cameras, device=args.device)
            
            bp  = True

        # if it % 100 == 0:
        #     num_reset = gaussians.halve_large_parallel_sigmas()
        #     print(f"Gaussians Scales Halved: {num_reset}")

        # periodic global opacity reset (commented out)
        # if it < opt.densify_until_iter and \
        #    (it % opt.opacity_reset_interval == 0 or
        #     (lpt.white_background and it == opt.densify_from_iter)):
        #     gaussians.reset_opacity()

        if it % 100 == 0 and it > opt.densify_until_iter:
            if it < opt.iterations - 100:
                gaussians.compute_3D_filter(cameras=all_cameras, device=args.device)

        with torch.no_grad():
            if it < opt.iterations:
                gaussians.optimizer.step()
                specular_mlp.optimizer.step()

                gaussians.optimizer.zero_grad(set_to_none=True)
                specular_mlp.optimizer.zero_grad()

                specular_mlp.update_learning_rate(it)

                if enable_uncertainty:
                    uncertainty_mlp.optimizer.step()
                    uncertainty_mlp.optimizer.zero_grad()
                    uncertainty_mlp.update_learning_rate(it)

        if it % 20 == 0:
            print(
                f"[{it:06d}] "
                f"huber {loss_h:.4f} (w {w_huber:.4f})  "
                f"seg {loss_seg:.4f} (w {w_seg:.4f})  "
                f"orient {loss_o:.4f} (w {w_orient:.4f})  "
                f"geom_fit {loss_geom_fit:.4f} (w {w_geom_fit:.4f})  "
                f"neigh {loss_nei:.4f} (w {w_nei:.4f})  "
                f"bend {loss_bend:.4f} (w {w_bend:.4f})  "
                f"sobel {loss_sobel:.4f} (w {w_sobel:.4f})  "
                f"head_col {loss_head_col:.4f} (w {w_head_col:.4f})  "
                f"gauss_head_col {loss_gauss_head_col:.4f} (w {w_gauss_head_col:.4f})  "
                f"local_len {loss_local_len:.4f} (w {w_local_len:.4f})  "
                f"color_var {loss_color_var:.4f} (w {w_color_var:.4f})  "
                f"opacity_var {loss_opacity_var:.4f} (w {w_opacity_var:.4f})  "
                f"asg_var {loss_asg_var:.4f} (w {w_asg_var:.4f})  "
                f"theta_l2 {loss_theta_l2:.4f} (w {w_theta_l2:.4f})  "
                f"beta_l2 {loss_beta_l2:.4f} (w {w_beta_l2:.4f})  "
                f"scale_reg {loss_scale_reg:.4f} (w {w_scale_reg:.4f})  "
                f"flame_rot_reg {loss_flame_rot_reg:.4f} (w {w_flame_rot_reg:.4f})  "
                f"flame_trans_reg {loss_flame_trans_reg:.4f} (w {w_flame_trans_reg:.4f})  "
                f"uncertainty_kl {loss_uncertainty_kl:.4f} (w {w_uncertainty_kl:.4f})  "
                f"dice {loss_dice:.4f} (w {w_dice:.4f})  "
                f"nz_frac {loss_nz_frac:.4f} (w {w_nz_frac:.4f})  "
                f"→ total {loss.item():.4f}     "
                f"→ Gaussian Count {gaussians.num_gaussians}     "
                f"→ Strand Count {gaussians.num_strands}     "
                f"→ Uncertainty Mean {uncertainty_vals.mean() if uncertainty_vals is not None else 0.0}     "
                f"→ Uncertainty Min {uncertainty_vals.min() if uncertainty_vals is not None else 0.0}     "
                f"→ Uncertainty Max {uncertainty_vals.max() if uncertainty_vals is not None else 0.0}     "
            )
            if _use_wandb:
                wandb.log({
                    "loss/total":            loss.item(),
                    "loss/huber":            loss_h.item(),
                    "loss/seg":              loss_seg.item(),
                    "loss/orient":           loss_o.item(),
                    "loss/geom_fit":         loss_geom_fit.item(),
                    "loss/neigh":            loss_nei.item(),
                    "loss/bend":             loss_bend.item(),
                    "loss/sobel":            loss_sobel.item(),
                    "loss/head_col":         loss_head_col.item(),
                    "loss/gauss_head_col":   loss_gauss_head_col.item(),
                    "loss/local_len":        loss_local_len.item(),
                    "loss/color_var":        loss_color_var.item(),
                    "loss/opacity_var":      loss_opacity_var.item(),
                    "loss/asg_var":          loss_asg_var.item(),
                    "loss/theta_l2":         loss_theta_l2.item(),
                    "loss/beta_l2":          loss_beta_l2.item(),
                    "loss/scale_reg":        loss_scale_reg.item(),
                    "loss/flame_rot_reg":    loss_flame_rot_reg.item(),
                    "loss/flame_trans_reg":  loss_flame_trans_reg.item(),
                    "loss/uncertainty_kl":   loss_uncertainty_kl.item(),
                    "loss/dice":             loss_dice.item(),
                    "loss/nz_frac":          loss_nz_frac.item(),
                    "iter":                  it,
                    "num_gaussians":         gaussians.num_gaussians,
                    "num_strands":           gaussians.num_strands,
                    "uncertianty mean":      uncertainty_vals.mean() if uncertainty_vals is not None else 0.0,
                    "uncertianty min":       uncertainty_vals.min() if uncertainty_vals is not None else 0.0,
                    "uncertianty max":       uncertainty_vals.max() if uncertainty_vals is not None else 0.0,
                }, step=it)

        if it % 250 == 0 or it == 1:
            gt_img    = gt_img * gt_alpha + bg_image * (1 - gt_alpha)
            canvas = make_side_by_side(gt_img, img_render, args.image_res)
            cv2.imwrite(os.path.join(train_dir, f"{it:06d}.png"), canvas[:, :, ::-1])
            seg_canvas = make_side_by_side(gt_alpha, img_segment, args.image_res)
            cv2.imwrite(os.path.join(train_dir, f"{it:06d}_seg.png"), seg_canvas[:, :, ::-1])
            seg_canvas = make_side_by_side(img_segment, render_pkg["gate"], args.image_res)
            cv2.imwrite(os.path.join(train_dir, f"{it:06d}_gate.png"), seg_canvas[:, :, ::-1])

        if it % 10000 == 0 or it == 1:
            export_strands_as_obj(
                strand_pts_can.detach(),
                os.path.join(train_dir, f"{it:06d}.obj")
            )

        if it % 2000 == 0 or it == 1:
            with torch.no_grad():
                dense_roots = perm.hair_roots.sample_scalp_mesh(10000, gaussians.roots).to(args.device)
                perm_out = perm(roots=dense_roots[None], theta=gaussians.theta, beta=gaussians.beta)

                strands = perm_out["strands"].position[0]
                sparse_strands = strands
                sparse_guide = guide_pts.detach()

                # guard against empty selections
                export_strands_as_obj(
                    sparse_strands,
                    os.path.join(train_dir, f"{it:06d}_dense.obj")
                )
            # if sparse_guide.numel() > 0:
            #     export_strands_as_obj(
            #         sparse_guide,
            #         os.path.join(train_dir, f"{it:06d}_guide.obj")
            #     )

            if _use_wandb:
                wandb.log({"preview": wandb.Image(canvas[:, :, ::-1])}, step=it)

        if it % 10000 == 0 or it == 1:
            torch.save(
                (gaussians.capture(), it, uniform_strand_color, enable_uncertainty, densify_count, densify_on_start, opt.densification_strand_interval, opt.densify_from_iter, opt.lambda_color_var, opt.lambda_opacity_var),
                os.path.join(model_dir, f"chkpnt_{it:06d}.pth")
            )
            specular_mlp.save_weights(model_dir, it)
            uncertainty_mlp.save_weights(model_dir, it)
            if lpt.learn_flame_rigid_offset:
                torch.save(rotation_offsets, os.path.join(model_dir, f"flame_rot_{it:06d}.pth"))
                torch.save(translation_offsets, os.path.join(model_dir, f"flame_trans_{it:06d}.pth"))
            print(f"\n[ITER {it}] Checkpoint saved.\n")

        cam.load2device("cpu")