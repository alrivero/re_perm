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
from scene import Scene_mica
from src.perm_deform_model import PermDeformModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import save_tensor_to_ply, export_strands_as_obj, export_strands_to_usd, save_tensor_to_obj, average_opacity_for_strand, average_opacity_per_strand, compute_occlusion_mask
from utils.loss_utils import (
    huber_loss,
    orientation_loss_v2_debug,
    strand_length_loss,
    neighbour_orientation_loss,
    outside_opacity_loss,
    orientation_match_strands_loss,
    oblong_shape_loss_from_strands_loss,
    length_consistency_loss_from_strands_loss,
    bending_loss,
    neighbor_scale_smoothness_loss,
    aligned_depth_loss,
    orientation_loss_kernel,
    head_collision_loss,
    sobel_loss,
    filtered_opacity_penalty,
    gaussian_head_collision_loss,
    local_length_consistency_loss,
    color_variance_loss_sh,
    opacity_variance_loss,
    asg_variance_loss,
    gaussian_scale_regularization_loss,
    triangle_scale_area_loss,
    sdf_hair_losses,
    sdf_contain_and_flow,
    l1_loss,
    ssim,
    HairDetailLoss,
    UncertaintyKLLoss, 
    alpha_blended_loss
)
from flame import FlameHead
from nphm.setup import setup_nphm_grid

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

def make_side_by_side(left: torch.Tensor, right: torch.Tensor, image_res: int) -> np.ndarray:
    left_np  = to_image_np(left)
    right_np = to_image_np(right)
    canvas = np.zeros((image_res, image_res * 2, 3), dtype=np.uint8)
    canvas[:, :image_res] = left_np
    canvas[:, image_res:] = right_np
    return canvas


def save_gate_tau_vis(
    gate_tensor: torch.Tensor,          # (..., 1, H, W) or (H, W)
    tau_tensor : torch.Tensor,          # same spatial shape as gate_tensor
    out_path   : Union[str, pathlib.Path],
    *,
    log_tau: bool = True,
    alpha_mask: Optional[torch.Tensor] = None,   # (H,W) mask of visible hair
    cmap_gate: str = "plasma",
    cmap_tau : str = "magma",
    dpi: int = 150,
):
    """
    Show gate (sigma(z) in [0,1]) and tau (variance or std-dev) side-by-side.

    Parameters
    ----------
    gate_tensor : torch.Tensor
        Either gate logits or probabilities.  If you pass logits, convert first
        with torch.sigmoid.
    tau_tensor : torch.Tensor
        Raw tau values (sigma, sigma², or log sigma).  Set log_tau accordingly.
    out_path : str | pathlib.Path
        Output file path; extension decides the format (e.g. “gate_tau_012.png”).
    log_tau : bool
        If True, log-scale tau before normalising to 0-1.
    alpha_mask : torch.Tensor | None
        Binary mask that marks foreground / background.  Background pixels are
        ignored for the min-max scaling so they stay neutral.
    """
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # -------- detach & squeeze to (H,W) -----------------------------------
    gate_map = gate_tensor.squeeze().float().detach().cpu()
    tau_map  = tau_tensor.squeeze().float().detach().cpu()

    if alpha_mask is not None:
        mask = alpha_mask.squeeze().bool().cpu()
    else:
        mask = torch.ones_like(gate_map, dtype=torch.bool)

    # -------- normalise gate to [0,1] -------------------------------------
    gate_fg   = gate_map[mask]
    gate_norm = (gate_fg - gate_fg.min()) / (gate_fg.max() - gate_fg.min() + 1e-12)
    gate_vis  = torch.zeros_like(gate_map)
    gate_vis[mask] = gate_norm

    # -------- normalise tau to [0,1] --------------------------------------
    tau_proc = torch.log10(tau_map.clamp_min(1e-6)) if log_tau else tau_map
    tau_fg   = tau_proc[mask]
    tau_norm = (tau_fg - tau_fg.min()) / (tau_fg.max() - tau_fg.min() + 1e-12)
    tau_vis  = torch.zeros_like(tau_map)
    tau_vis[mask] = tau_norm

    # -------- plot --------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=dpi, constrained_layout=True)

    ax[0].imshow(gate_vis.numpy(), cmap=cmap_gate)
    ax[0].set_title("Gate σ(z)")
    ax[0].axis("off")

    ax[1].imshow(tau_vis.numpy(), cmap=cmap_tau)
    ax[1].set_title("Tau (log)" if log_tau else "Tau")
    ax[1].axis("off")

    fig.suptitle(out_path.name, fontsize=9)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

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
    parser = argparse.ArgumentParser("perm-fitting")

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--idname", type=str, default="id1_25")
    parser.add_argument("--image_res", type=int, default=720)
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
    mica_datadir = os.path.join(data_dir, "track_out", args.idname)
    log_dir      = os.path.join(data_dir, "log")
    train_dir    = os.path.join(log_dir, "train")
    model_dir    = os.path.join(log_dir, "ckpt")

    scene = Scene_mica(
        data_dir, mica_datadir,
        train_type=0,
        white_background=lpt.white_background,
        device=args.device
    )
    all_cameras = scene.getCameras().copy()
    viewpoint_stack = None

    # sample_pow = 0
    # sample_idxs = [0]

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    scalp_mask = pickle.load(
        open(
            "/home/alrivero/ENPC/re_perm/flame/assets/FLAME_masks.pkl",
            "rb"
        ),
        encoding="latin1"
    )["scalp"]


    # 1) load your mesh (only once)
    flame = FlameHead(
        300, 
        100, 
        add_teeth=True,
        remove_lip_inside=False,
        face_clusters=("skin", "hair", "boundary", "lips_tight", "teeth", "sclerae", "irises"),
    ).to(args.device)

    sb = 1.0
    perm = Perm(
        lpt.perm_path,
        lpt.obj_head_path,
        scalp_vertex_idxs=scalp_mask,
        scalp_bounds=[0.1870, 0.8018, 0.4011, 0.8047],
        mesh_scale=100.0
    ).to(args.device)
    perm = perm.eval()

    pseudo_roots = perm.hair_roots.load_txt(lpt.loaded_roots_path)[0]

    start_hair_style = None
    if lpt.emp_hair_path:
        start_hair_style = np.load(lpt.emp_hair_path)

    if os.path.exists(lpt.cached_roots_path):
        cached_roots = torch.load(lpt.cached_roots_path)
    else:
        cached_roots = None

    gaussians = GaussianPerm(perm, pseudo_roots, start_hair_style, lpt.sh_degree, lpt.asg_degree, cached_roots=cached_roots).to(args.device)
    gaussians.roots = gaussians.roots.to(args.device)

    extra_parameters = None
    if lpt.learn_flame_rigid_offset:
        rotation_offsets = torch.tensor([0.0, 0.0, 0.0],
                            dtype=torch.float32,
                            device=args.device)
        rotation_offsets = rotation_offsets.unsqueeze(0).repeat(len(scene.getCameras()), 1).requires_grad_()
        translation_offsets = torch.tensor([0.0, 0.0, 0.0],
                                dtype=torch.float32,
                                device=args.device)
        translation_offsets = translation_offsets.unsqueeze(0).repeat(len(scene.getCameras()), 1).requires_grad_()
        extra_parameters = [
            {"params": [rotation_offsets], "lr": 0.0001, "name": "flame_rot"},
            {"params": [translation_offsets], "lr": 0.0001, "name": "flame_trans"}
        ]

        torch.save(rotation_offsets, os.path.join(model_dir, f"flame_rot.pth"))
    gaussians.training_setup(opt, extra_parameters=extra_parameters)

    # Set up NPHM SDF
    nphm_config = yaml.safe_load(open(args.nphm_config_path, 'r'))

    loaded_dict = torch.load(args.dict_loaded_nphm_path)
    anchors_path = "nphm/assets/anchors_65.npy"
    anchors = torch.from_numpy(np.load(anchors_path)).float().unsqueeze(0).unsqueeze(0).to(args.device)
    geo_encoding = torch.from_numpy(np.load(args.geo_nphm_path)).to(args.device)
    app_encoding = torch.from_numpy(np.load(args.app_nphm_path)).to(args.device)
    exp_encoding = torch.from_numpy(np.load(args.xp_nphm_path)).to(args.device)
    latent_code = {
        'geo': geo_encoding,
        'app': app_encoding,
        'exp': exp_encoding[None]
    }

    # compute original AABB
    roots = gaussians.roots
    aabb_min, _ = roots.min(dim=0)
    aabb_min /= 100
    aabb_max, _ = roots.max(dim=0)
    aabb_max /= 100

    # compute rescaled AABB
    S = 2.5
    center       = 0.5 * (aabb_min + aabb_max)
    half_sizes   = 0.5 * (aabb_max - aabb_min) * S   
    half_len     = half_sizes.max()
    center[1] = aabb_max[1]

    aabb_min_cube = center - half_len
    aabb_max_cube = center + half_len

    nphm_grid = None
    if os.path.exists("/data/add_disk0/alrivero/imagine_data/arjit/nphm_octree_full.pkl"):
        nphm_grid = torch.load("/data/add_disk0/alrivero/imagine_data/arjit/nphm_octree_full.pkl", map_location=args.device)
        nphm_grid.pyramids = nphm_grid.pyramids.cpu()
    else:
        nphm_grid = setup_nphm_grid(
            nphm_config,
            loaded_dict,
            latent_code,
            anchors,
            aabb_min_cube,
            aabb_max_cube,
            4.0,
            256,
            args.device
        )
        torch.save(nphm_grid, "/data/add_disk0/alrivero/imagine_data/arjit/nphm_octree_full.pkl")
        
    torch.cuda.empty_cache()
    torch.save(gaussians.roots, lpt.cached_roots_path)

    # --- freeze during warm-up -------------------------------------
    toggle_beta_trainable(gaussians, False, opt)

    deform_model = PermDeformModel(perm, flame, args.device).to(args.device)
    deform_model.training_setup()

    specular_mlp = SpecularModel()
    specular_mlp.specular = specular_mlp.specular.to(args.device)
    specular_mlp.train_setting(opt)

    first_iter = 0
    if args.start_checkpoint:
        m_params, g_params, first_iter = torch.load(args.start_checkpoint)
        deform_model.restore(m_params)
        gaussians.restore(g_params, opt, extra_parameters=extra_parameters)
        specular_mlp.load_weights(model_dir, iteration=first_iter)
        first_iter -= 1

        if lpt.learn_flame_rigid_offset:
            rotation_offsets = torch.load("/data/add_disk0/alrivero/imagine_data/arjit/log/ckpt/flame_rot_070000.pth")
            translation_offsets = torch.load("/data/add_disk0/alrivero/imagine_data/arjit/log/ckpt/flame_trans_070000.pth")

    bg_color = [1, 1, 1] if lpt.white_background else [0, 1, 0]
    bg_image = torch.zeros((3, args.image_res, args.image_res), device=args.device)
    if lpt.white_background:
        bg_image[:] = 1
    else:
        bg_image[1] = 1
    background = torch.tensor(bg_color, dtype=torch.float32, device=args.device)

    save_tensor_to_obj(gaussians.roots / 100, os.path.join(train_dir, "roots.obj"))

    hair_photo_loss = HairDetailLoss(
        warmup_iters=8000,
        fade_iters=12000,
        polish_iters=5000,
        blur=True,
        device=args.device
    )

    kl_loss = UncertaintyKLLoss(
        beta_start = 1e-2,        # off for the first few k steps
        beta_final = 4e-2,       # weighted-KL ≈ 5 at the start of the ramp
        t_start    = 8_000,      # begin ramp here
        t_end      = 30_000,     # reach full strength well before 50k
        sigma_z    = 1.0,
        mu_tau     = math.log(0.02),
        sigma_tau  = 0.5,
    )

    uniform_strand_color = True
    enable_uncertainty = True

    for it in range(first_iter + 1, opt.iterations + 1):

        if it % 500 == 0:
            gaussians.oneupSHdegree()

        if it == opt.theta_warmup + 1:
            toggle_beta_trainable(gaussians, True, opt)

        # if it % 3000 == 0 and it != first_iter + 1:
        #     sample_pow += 1
        #     sample_idxs = list(range(0, len(all_cameras), int(len(all_cameras) / min(len(all_cameras), (4 ** sample_pow)))))

        if not viewpoint_stack:
            viewpoint_stack = scene.getCameras().copy().tolist()
            random.shuffle(viewpoint_stack)
        cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack) - 1))
        cam.load2device(args.device)

        enable_flame_offsets = lpt.learn_flame_rigid_offset and it >= 10000
        flame_rot = rotation_offsets[cam.uid][None] if enable_flame_offsets else torch.tensor([[0.0, 0.0, 0.0]]).to(args.device)
        flame_trans = translation_offsets[cam.uid][None] if enable_flame_offsets else torch.tensor([[0.0, 0.0, 0.0]]).to(args.device)
        
        codedict = {
            "R":      torch.tensor(cam.R, device=args.device),
            "T":      torch.tensor(cam.T, device=args.device),
            "roots":  gaussians.get_roots_xyz[None],
            "theta":  gaussians.theta,
            "beta":   gaussians.beta,
            "expr":   cam.exp_param,
            "shape":   cam.shape_param,
            "eyes_pose":   cam.eyes_pose,
            "jaw_pose":   cam.jaw_pose,
            "neck_pose":   cam.neck_pose,
            "root_pose":   torch.zeros_like(flame_rot),
            "translation":   torch.zeros_like(flame_trans)
        }

        flame_verts, flame_faces, flame_normals = flame(
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

        verts_final, guide_final, verts_final_def, guide_final_def, rot_delta, scale_coef = deform_model.decode(gaussians, codedict)
        strand_pts = verts_final_def.reshape(gaussians.num_strands, STRAND_VERTEX_COUNT, 3)
        strand_pts_can = verts_final.reshape(gaussians.num_strands, STRAND_VERTEX_COUNT, 3)

        tangents = gaussians.update_xyz_rot_scale(strand_pts, rot_delta, scale_coef)

        if it == first_iter + 1:
            gaussians.compute_3D_filter(cameras=all_cameras, device=args.device)

        if it >= 3000:
            dir_pp = (gaussians.get_xyz - cam.camera_center.repeat(gaussians.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            spec_color = specular_mlp.step(gaussians.get_asg_features, dir_pp_normalized, tangents)
        else:
            spec_color = 0.0

        occ_mask, _ = compute_occlusion_mask(gaussians, cam, flame_verts, flame_faces)

        render_pkg = render(cam, gaussians, ppt, background, kernel_size=lpt.kernel_size, occ_mask=occ_mask, spec_color=spec_color)
        img_render   = render_pkg["render"]
        img_segment  = render_pkg["segment"]

        gt_img    = cam.original_image
        alpha     = cam.hair_mask
        orient    = cam.hair_orient
        depth_gt  = cam.depth_map
        gt_img    = gt_img * alpha + bg_image * (1 - alpha)

        
        if enable_uncertainty:
            loss_h = hair_photo_loss(img_render, gt_img, it, gate_map=render_pkg["gate"], tau_map=render_pkg["tau"])
        else:
            loss_h = hair_photo_loss(img_render, gt_img, it)

        loss_seg = alpha_blended_loss(img_segment, alpha)

        guide_pts = guide_final.reshape(-1, STRAND_VERTEX_COUNT, 3)

        loss_o                    = orientation_loss_v2_debug(cam, gaussians, alpha, orient, occ_mask)
        loss_sdf_contain, loss_sdf_flow = sdf_contain_and_flow(nphm_grid, gaussians, strand_pts_can)
        loss_nei                  = neighbour_orientation_loss(strand_pts[:, :STRAND_VERTEX_COUNT, :], gaussians.neighbor_idx)
        loss_bend                 = bending_loss(strand_pts)
        loss_sobel                = triangle_scale_area_loss(gaussians)
        loss_head_col             = head_collision_loss(strand_pts, flame_verts[0], flame_normals[0])
        loss_gauss_head_col       = gaussian_head_collision_loss(gaussians, flame_verts[0], flame_normals[0])
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

        if enable_uncertainty:
            loss_uncertainty_kl = kl_loss(gaussians, it)
        else:
            loss_uncertainty_kl = torch.tensor(0.0)

        if False:
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
        lambda_sdf_contain    = opt.lambda_sdf_contain
        lambda_sdf_flow       = opt.lambda_sdf_flow
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

        w_huber           = lambda_huber                * loss_h.item()
        w_nei             = lambda_nei                  * loss_nei.item()
        w_orient          = lambda_orient               * loss_o.item()
        w_seg             = lambda_seg                  * loss_seg.item()
        w_sdf_contain     = lambda_sdf_contain          * loss_sdf_contain.item()
        w_sdf_flow        = lambda_sdf_flow             * loss_sdf_flow.item()
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
        w_uncertainty_kl = lambda_uncertainty_kl      *  loss_uncertainty_kl.item()

        loss = (
            lambda_huber           * loss_h +
            lambda_seg             * loss_seg +
            lambda_orient          * loss_o +
            lambda_sdf_contain     * loss_sdf_contain +
            lambda_sdf_flow        * loss_sdf_flow +
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
            lambda_uncertainty_kl  * loss_uncertainty_kl
        )

        gaussians.update_learning_rate(it)
        loss.backward()

        # Densification
        radii                  = render_pkg["radii"]                 # (M,) for renderer
        visibility_filter      = render_pkg["visibility_filter"]     # (M,) for renderer
        viewspace_point_tensor = render_pkg["viewspace_points"]

        # track running ∥∇xy∥ statistics
        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, occ_mask)

        # densify / prune within schedule window
        if it < opt.densify_until_iter and it > opt.densify_from_iter \
                                        and it % opt.densification_interval == 0:
            if occ_mask is not None:
                with torch.no_grad():
                    radii = render(cam, gaussians, ppt, background, kernel_size=lpt.kernel_size, spec_color=spec_color)["radii"]
                                        
            size_threshold = 20 if it > opt.opacity_reset_interval else None
            num_clone, num_split = gaussians.densify_and_prune(
                opt.densify_grad_threshold,            # gradient criterion
                0.05,                                 # min-opacity for pruning
                scene.cameras_extent,                  # scene extent
                size_threshold,                        # screen-size pruning
                radii                                  # 2-D radii per Gaussian
            )
            print(f"\n[ITER {it}] Gaussians Cloned: {num_clone} Gaussians Split {num_split}.\n")
            print(f"\n[ITER {it}] Scale Min: {gaussians.get_scaling[:, 1].min()}, Scale Max: {gaussians.get_scaling[:, 1].max()}, Scale Median: {gaussians.get_scaling[:, 1].median()}\n")
            print(f"\n[ITER {it}] 5th Quantile: {gaussians.get_scaling[:, 1].quantile(0.05)}, 10th Quantile: {gaussians.get_scaling[:, 1].quantile(0.10)}, 15th Quantile: {gaussians.get_scaling[:, 1].quantile(0.15)}\n")

            if num_clone + num_split > 0:
                gaussians.compute_3D_filter(cameras=all_cameras, device=args.device)
        if it < opt.densify_strands_until_iter and it > opt.densify_strands_from_iter \
                                        and it % opt.densification_strand_interval == 0:
            torch.save(
                (deform_model.capture(), gaussians.capture(), it),
                os.path.join(model_dir, f"chkpnt_{it:06d}_pre_dense.pth")
            )
            specular_mlp.save_weights(model_dir, it)
            print(f"\n[ITER {it}] Pre-Densification Checkpoint saved.\n")

            new_roots, new_radii = perm.hair_roots.densify_scalp_hex(gaussians.roots)
            while new_roots.shape[0] <= gaussians.roots.shape[0]:
                new_roots, new_radii = perm.hair_roots.densify_scalp_hex(gaussians.roots)

            gaussians.reset_gaussians_to_new_roots(new_roots, new_radii, uniform_strand_color=True)
            gaussians.compute_3D_filter(cameras=all_cameras, device=args.device)

            gaussians.training_setup(opt, extra_parameters=extra_parameters)
            save_tensor_to_obj(gaussians.roots / 100, os.path.join(train_dir, f"roots_{it}.obj"))

            opt.densification_strand_interval *= 4
            opt.densify_from_iter = 500 + it
            opt.lambda_color_var *= 10.0

            uniform_strand_color = True
            enable_uncertainty = True

        if it % 100 == 0:
            num_reset = gaussians.halve_large_parallel_sigmas()
            print(f"Gaussians Scales Halved: {num_reset}")

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
                deform_model.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform_model.optimizer.zero_grad(set_to_none=True)
                specular_mlp.optimizer.step()
                specular_mlp.optimizer.zero_grad()
                specular_mlp.update_learning_rate(it)

        if it % 20 == 0:
            print(
                f"[{it:06d}] "
                f"huber {loss_h:.4f} (w {w_huber:.4f})  "
                f"seg {loss_seg:.4f} (w {w_seg:.4f})  "
                f"orient {loss_o:.4f} (w {w_orient:.4f})  "
                f"sdf_contain {loss_sdf_contain:.4f} (w {w_sdf_contain:.4f})  "
                f"sdf_flow {loss_sdf_flow:.4f} (w {w_sdf_flow:.4f})  "
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
                f"→ total {loss.item():.4f}     "
                f"→ Gaussian Count {gaussians.num_gaussians}     "
                f"→ Strand Count {gaussians.num_strands}     "
            )
            if _use_wandb:
                wandb.log({
                    "loss/total":            loss.item(),
                    "loss/huber":            loss_h.item(),
                    "loss/seg":              loss_seg.item(),
                    "loss/orient":           loss_o.item(),
                    "loss/sdf_contain":      loss_sdf_contain.item(),
                    "loss/sdf_flow":         loss_sdf_flow.item(),
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
                    "iter":                  it,
                    "num_gaussians":         gaussians.num_gaussians,
                    "num_strands":           gaussians.num_strands,
                }, step=it)

        if it % 500 == 0 or it == 1:
            canvas = make_side_by_side(gt_img, img_render, args.image_res)
            cv2.imwrite(os.path.join(train_dir, f"{it:06d}.png"), canvas[:, :, ::-1])
            seg_canvas = make_side_by_side(alpha, img_segment, args.image_res)
            cv2.imwrite(os.path.join(train_dir, f"{it:06d}_seg.png"), seg_canvas[:, :, ::-1])
            save_gate_tau_vis(render_pkg["gate"], render_pkg["tau"], os.path.join(train_dir, f"{it:06d}_gate_tau.png"))

        if it % 10000 == 0 or it == 1:
            export_strands_as_obj(
                strand_pts_can.detach(),
                os.path.join(train_dir, f"{it:06d}.obj")
            )

        if it % 1000 == 0 or it == 1:
            with torch.no_grad():
                dense_roots = perm.hair_roots.sample_scalp_mesh(10000, pseudo_roots).to(args.device)
                perm_out = perm(roots=dense_roots[None], theta=gaussians.theta, beta=gaussians.beta)

                strands = perm_out["strands"].position[0]
                sparse_strands = strands / 100.0
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
                (deform_model.capture(), gaussians.capture(), it),
                os.path.join(model_dir, f"chkpnt_{it:06d}.pth")
            )
            specular_mlp.save_weights(model_dir, it)
            if lpt.learn_flame_rigid_offset:
                torch.save(rotation_offsets, os.path.join(model_dir, f"flame_rot_{it:06d}.pth"))
                torch.save(translation_offsets, os.path.join(model_dir, f"flame_trans_{it:06d}.pth"))
            print(f"\n[ITER {it}] Checkpoint saved.\n")

        cam.load2device("cpu")