import os, sys, random, argparse, pickle, cv2
import numpy as np
import torch
import torch.nn as nn
import lpips
import datetime as dt
import debug

try:
    import wandb
    _use_wandb = True
except ModuleNotFoundError:
    print("[wandb] not found – continuing without online logging.")
    _use_wandb = False

# --------------------------------------------------------------------- #
# project imports
# --------------------------------------------------------------------- #
from hair.hair_models import Perm
from scene.gaussian_perm import GaussianPerm
from scene import Scene_mica
from src.perm_deform_model import PermDeformModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import save_tensor_to_ply, export_strands_as_obj, export_strands_to_usd
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
    strand_repulsion_loss,
    sobel_loss,
    filtered_opacity_penalty
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

def toggle_beta_trainable(gaussians, trainable: bool):
    """
    Sets β’s learning-rate to 0 and disables gradients when
    `trainable=False`; restores both when `True`.
    """
    for pg in gaussians.optimizer.param_groups:
        if pg.get("name") == "beta":
            pg["lr"] = gaussians._beta_base_lr if trainable else 0.0

    gaussians.beta.requires_grad_(trainable)

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
    parser.add_argument("--image_res", type=int, default=512)
    parser.add_argument("--start_checkpoint", type=str, default=None)

    args = parser.parse_args(sys.argv[1:])
    args.device = "cuda"

    set_random_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)

    lpt = lp.extract(args)
    opt = op.extract(args)
    ppt = pp.extract(args)

    if _use_wandb:
        run_name = f"{args.idname}—{dt.datetime.now().strftime('%b-%d-%Y_%Hh%Mm')}"
        wandb.init(project="re_perm", name=run_name,
                   config={**args.__dict__, **opt.__dict__, **ppt.__dict__, **lpt.__dict__})

    data_dir     = os.path.join(args.source_path, args.idname)
    mica_datadir = os.path.join(data_dir, "track_out", args.idname)
    log_dir      = os.path.join(data_dir, "log")
    train_dir    = os.path.join(log_dir, "train")
    model_dir    = os.path.join(log_dir, "ckpt")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    scalp_mask = pickle.load(
        open("/home/alrivero/ENPC/re_perm/flame/FLAME_masks/FLAME_masks.pkl", "rb"),
        encoding="latin1"
    )["scalp"]

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

    gaussians = GaussianPerm(perm, pseudo_roots, start_hair_style, lpt.sh_degree).to(args.device)
    gaussians.roots = gaussians.roots.to(args.device)
    gaussians.training_setup(opt)

    # --- freeze during warm‑up -------------------------------------
    toggle_beta_trainable(gaussians, trainable=False)   # lr = 0

    deform_model = PermDeformModel(perm, args.device).to(args.device)
    deform_model.training_setup()

    first_iter = 0
    if args.start_checkpoint:
        m_params, g_params, first_iter = torch.load(args.start_checkpoint)
        deform_model.restore(m_params)
        gaussians.restore(g_params, opt)

    bg_color = [1, 1, 1] if lpt.white_background else [0, 1, 0]
    bg_image = torch.zeros((3, args.image_res, args.image_res), device=args.device)
    if lpt.white_background:
        bg_image[:] = 1
    else:
        bg_image[1] = 1
    background = torch.tensor(bg_color, dtype=torch.float32, device=args.device)

    scene = Scene_mica(
        data_dir, mica_datadir,
        train_type=0,
        white_background=lpt.white_background,
        device=args.device
    )

    all_cameras = scene.getCameras().copy()
    viewpoint_stack = None
    for it in range(first_iter + 1, opt.iterations + 1):
        if it % 500 == 0:
            gaussians.oneupSHdegree()

        if it == opt.theta_warmup + 1:
            toggle_beta_trainable(gaussians, trainable=True)   # lr = 0

        if not viewpoint_stack:
            viewpoint_stack = scene.getCameras().copy()
            random.shuffle(viewpoint_stack)
        cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack) - 1))
        cam.load2device(args.device)

        codedict = {
            "R":     torch.tensor(cam.R, device=args.device),
            "T":     torch.tensor(cam.T, device=args.device),
            "roots": gaussians.get_roots_xyz[None],
            "theta": gaussians.theta,
            "beta":  gaussians.beta,
        }

        verts_final, guide_final, rot_delta, scale_coef = deform_model.decode(gaussians, codedict)
        strand_pts = verts_final.reshape(gaussians.num_strands, STRAND_VERTEX_COUNT, 3)
        gaussians.update_xyz_rot_scale(strand_pts, rot_delta, scale_coef)

        if it == first_iter + 1:
            gaussians.compute_3D_filter(cameras=all_cameras, device=args.device)

        render_pkg = render(cam, gaussians, ppt, background, kernel_size=lpt.kernel_size)
        img_render   = render_pkg["render"]
        img_segment  = render_pkg["segment"]
        # depth_pred   = render_pkg["depth"][0]

        gt_img  = cam.original_image
        alpha   = cam.hair_mask
        orient  = cam.hair_orient
        depth_gt   = cam.depth_map
        gt_img  = gt_img * alpha + bg_image * (1 - alpha)

        loss_h   = huber_loss(img_render, gt_img, alpha, delta=0.1, reduction="mean")
        if it <= opt.theta_warmup:
            loss_seg = huber_loss(img_segment, alpha, alpha, delta=0.1, reduction="mean")
        else:
            loss_seg = huber_loss(img_segment, (alpha == 1).float(), torch.ones_like(alpha), delta=0.1, reduction="mean")

        guide_pts = guide_final.reshape(-1, STRAND_VERTEX_COUNT, 3)

        loss_o             = orientation_loss_v2_debug(cam, gaussians, alpha, orient)
        loss_len           = strand_length_loss(strand_pts, L_max=opt.max_strand_len, delta=opt.delta_strand_len)
        loss_nei           = neighbour_orientation_loss(strand_pts[:, :STRAND_VERTEX_COUNT, :], k=opt.k_neigh)
        loss_bend          = bending_loss(strand_pts)
        loss_depth         = torch.tensor(0.0).to(args.device) # No depth
        loss_head_col      = sobel_loss(img_render, gt_img, alpha) # head_collision_loss(strand_pts, gaussians.get_roots_xyz / 100, margin=0.0015)
        loss_strand_rep    = filtered_opacity_penalty(gaussians) # strand_repulsion_loss(strand_pts, k=8, step=2, safe_dist=0.0025)

        mu_theta = gaussians.theta_start.clone().to(args.device)  # no grad
        mu_beta  = gaussians.beta_start.clone().to(args.device)
        loss_theta_l2 = (gaussians.theta - mu_theta).pow(2).sum()
        loss_beta_l2  = (gaussians.beta  - mu_beta).pow(2).sum()

        lambda_huber        = opt.lambda_huber
        lambda_seg          = opt.lambda_seg
        lambda_depth        = opt.lambda_depth
        lambda_neigh        = opt.lambda_neigh
        lambda_orient       = opt.lambda_orient
        lambda_len          = opt.lambda_len
        lambda_bend         = opt.lambda_bend
        lambda_head_col     = opt.lambda_head_col
        lambda_strand_rep   = opt.lambda_strand_rep
        lambda_theta_l2     = opt.lambda_theta_l2
        lambda_beta_l2      = opt.lambda_beta_l2

        w_huber         = lambda_huber        * loss_h.item()
        w_neigh         = lambda_neigh        * loss_nei.item()
        w_orient        = lambda_orient       * loss_o.item()
        w_seg           = lambda_seg          * loss_seg.item()
        w_len           = lambda_len          * loss_len.item()
        w_bend          = lambda_bend         * loss_bend.item()
        w_depth         = lambda_depth        * loss_depth.item()
        w_head_col      = lambda_head_col     * loss_head_col.item()
        w_strand_rep    = lambda_strand_rep   * loss_strand_rep.item()
        w_theta_l2      = lambda_theta_l2     * loss_theta_l2.item()
        w_beta_l2       = lambda_beta_l2      * loss_beta_l2.item()

        loss = (
            lambda_huber        * loss_h +
            lambda_seg          * loss_seg +
            lambda_orient       * loss_o +
            lambda_len          * loss_len +
            lambda_neigh        * loss_nei +
            lambda_bend         * loss_bend +
            lambda_depth        * loss_depth +
            lambda_head_col     * loss_head_col +
            lambda_strand_rep   * loss_strand_rep +
            lambda_theta_l2     * loss_theta_l2 +
            lambda_beta_l2      * loss_beta_l2
        )

        gaussians.update_learning_rate(it)
        loss.backward()

        # Densification
        radii             = render_pkg["radii"]            #  (M,)  – add to renderer
        visibility_filter = render_pkg["visibility_filter"]       #  (M,)  – add to renderer
        viewspace_point_tensor = render_pkg["viewspace_points"]

        # track running ∥∇xy∥ statistics
        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

        # densify / prune as long as we’re in the schedule window
        if it < opt.densify_until_iter and it > opt.densify_from_iter \
                                        and it % opt.densification_interval == 0:
            size_threshold = 20 if it > opt.opacity_reset_interval else None
            gaussians.densify_and_prune(
                opt.densify_grad_threshold,           # gradient criterion
                0.005,                                # min-opacity for pruning
                scene.cameras_extent,                 # scene extent
                size_threshold,                       # screen-size pruning
                radii                                  # 2-D radii per Gaussian
            )
            gaussians.compute_3D_filter(cameras=all_cameras, device=args.device)

        # periodic global opacity reset
        # if it < opt.densify_until_iter and \
        #     (it % opt.opacity_reset_interval == 0 or
        #         (lpt.white_background and it == opt.densify_from_iter)):
        #         gaussians.reset_opacity()

        if it % 100 == 0 and it > opt.densify_until_iter:
            if it < opt.iterations - 100:
                gaussians.compute_3D_filter(cameras=all_cameras, device=args.device)

        with torch.no_grad():
            if it < opt.iterations:
                gaussians.optimizer.step()
                deform_model.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform_model.optimizer.zero_grad(set_to_none=True)

        if it % 20 == 0:
            print(
                f"[{it:06d}] "
                f"huber {loss_h:.4f} (w {w_huber:.4f})  "
                f"seg {loss_seg:.4f} (w {w_seg:.4f})  "
                f"orient {loss_o:.4f} (w {w_orient:.4f})  "
                f"len {loss_len:.4f} (w {w_len:.4f})  "
                f"neigh {loss_nei:.4f} (w {w_neigh:.4f})  "
                f"bend {loss_bend:.4f} (w {w_bend:.4f})  "
                f"depth {loss_depth:.4f} (w {w_depth:.4f})  "
                f"head_col {loss_head_col:.4f} (w {w_head_col:.4f})  "
                f"strand_rep {loss_strand_rep:.4f} (w {w_strand_rep:.4f})  "
                f"theta_l2 {loss_theta_l2:.4f} (w {w_theta_l2:.4f})  "
                f"beta_l2 {loss_beta_l2:.4f} (w {w_beta_l2:.4f})  "
                f"→ total {loss.item():.4f}     "
                f"→ Gaussian Count {gaussians.num_gaussians}     "
            )
            if _use_wandb:
                wandb.log({
                    "loss/total":        loss.item(),
                    "loss/huber":        loss_h.item(),
                    "loss/seg":          loss_seg.item(),
                    "loss/orient":       loss_o.item(),
                    "loss/len":          loss_len.item(),
                    "loss/neigh":        loss_nei.item(),
                    "loss/bend":         loss_bend.item(),
                    "loss/depth":        loss_depth.item(),
                    "loss/head_col":     loss_head_col.item(),
                    "loss/strand_rep":   loss_strand_rep.item(),
                    "loss/theta_l2":     loss_theta_l2.item(),
                    "loss/beta_l2":      loss_beta_l2.item(),
                    "iter":              it,
                    "num_gaussians": gaussians.num_gaussians
                }, step=it)

        if it % 1000 == 0 or it == 1:
            canvas = make_side_by_side(gt_img, img_render, args.image_res)
            cv2.imwrite(os.path.join(train_dir, f"{it:06d}.png"), canvas[:, :, ::-1])
            seg_canvas = make_side_by_side(alpha, img_segment, args.image_res)
            cv2.imwrite(os.path.join(train_dir, f"{it:06d}_seg.png"), seg_canvas[:, :, ::-1])
            # depth_canvas = make_side_by_side(depth_gt[None].expand(3, -1, -1), (depth_rescaled * (depth_gt > 0.0).float())[None].expand(3, -1, -1), args.image_res)
            # cv2.imwrite(os.path.join(train_dir, f"{it:06d}_depth.png"), depth_canvas[:, :, ::-1])

        if it % 2000 == 0 or it == 1:
            with torch.no_grad():
                dense_roots = perm.hair_roots.sample_scalp_mesh(10000, pseudo_roots).to(args.device)
                perm_out = perm(roots=dense_roots[None], theta=gaussians.theta, beta=gaussians.beta)

                strands = perm_out["strands"].position[0]
                sparse_strands = strands / 100.0
                sparse_guide = guide_pts.detach()

            # guard against empty selections
            if sparse_strands.numel() > 0:
                export_strands_as_obj(
                    sparse_strands,
                    os.path.join(train_dir, f"{it:06d}.obj")
                )
            if sparse_guide.numel() > 0:
                export_strands_as_obj(
                    sparse_guide,
                    os.path.join(train_dir, f"guide_{it:06d}.obj")
                )
            # export_strands_to_usd(f"{train_dir}/{it:06d}.usda", strand_pts, gaussians)

            if _use_wandb:
                wandb.log({"preview": wandb.Image(canvas[:, :, ::-1])}, step=it)

        if it % 5000 == 0 or it == 1:
            torch.save((deform_model.capture(), gaussians.capture(), it),
                       os.path.join(model_dir, f"chkpnt_{it:06d}.pth"))
            print(f"\n[ITER {it}] Checkpoint saved.\n")

        cam.load2device("cpu")