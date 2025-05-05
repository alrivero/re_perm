import os, sys, random, argparse, pickle, cv2
import numpy as np
import torch
import torch.nn as nn
import lpips

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
from utils.general_utils import save_tensor_to_ply
from utils.loss_utils import (
    huber_loss,
    orientation_loss,
    strand_length_loss,
    neighbour_orientation_loss,
)

STRAND_VERTEX_COUNT = 25    # same as in GaussianPerm


# --------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------- #
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser("perm‑fitting")

    # default groups (unchanged)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # experiment‑specific flags (unchanged)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--idname", type=str, default="id1_25")
    parser.add_argument("--image_res", type=int, default=512)
    parser.add_argument("--start_checkpoint", type=str, default=None)

    args = parser.parse_args(sys.argv[1:])
    args.device = "cuda"

    # ------------------------------------------------------------------ #
    #  initialise
    # ------------------------------------------------------------------ #
    set_random_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)

    lpt = lp.extract(args)
    opt = op.extract(args)
    ppt = pp.extract(args)

    # --------------------- wandb init --------------------------------- #
    if _use_wandb:
        wandb.init(
            project="re_perm",
            name=args.idname,
            config={**args.__dict__,
                    **opt.__dict__},
        )

    # ------------------------- data paths ----------------------------- #
    data_dir       = os.path.join(args.source_path, args.idname)
    mica_datadir   = os.path.join(data_dir, "track_out", args.idname)
    log_dir        = os.path.join(data_dir, "log")
    train_dir      = os.path.join(log_dir, "train")
    model_dir      = os.path.join(log_dir, "ckpt")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # ------------------------- FLAME scalp mask ----------------------- #
    scalp_mask = pickle.load(open(
        "/home/alrivero/ENPC/re_perm/flame/FLAME_masks/FLAME_masks.pkl", "rb"),
        encoding="latin1")["scalp"]

    # ------------------------- PERM mesh ------------------------------ #
    perm = Perm(
        lpt.perm_path,
        lpt.obj_head_path,
        scalp_vertex_idxs=scalp_mask,
        scalp_bounds=[0.1870, 0.8018, 0.4011, 0.8047],
        mesh_scale=100.0).to(args.device)

    pseudo_roots = perm.hair_roots.load_txt(lpt.loaded_roots_path)[0]

    start_hair_style = None
    if lpt.emp_hair_path:
        start_hair_style = np.load(lpt.emp_hair_path)

    gaussians = GaussianPerm(
        perm,
        pseudo_roots,
        start_hair_style,
        lpt.sh_degree).to(args.device)
    gaussians.roots = gaussians.roots.to(args.device)
    gaussians.training_setup(opt)

    # ------------------------- deformation model ---------------------- #
    deform_model = PermDeformModel(perm, args.device).to(args.device)
    deform_model.training_setup()

    # restore if checkpoint provided
    first_iter = 0
    if args.start_checkpoint:
        m_params, g_params, first_iter = torch.load(args.start_checkpoint)
        deform_model.restore(m_params)
        gaussians.restore(g_params, opt)

    # ------------------------- background / scene -------------------- #
    bg_color = [1, 1, 1] if lpt.white_background else [0, 1, 0]
    bg_image = torch.zeros((3, args.image_res, args.image_res),
                           device=args.device)
    if lpt.white_background:
        bg_image[:] = 1
    else:
        bg_image[1] = 1
    background = torch.tensor(bg_color, dtype=torch.float32, device=args.device)

    scene = Scene_mica(
        data_dir, mica_datadir,
        train_type=0,
        white_background=lpt.white_background,
        device=args.device)

    # ------------------------- main loop ----------------------------- #
    viewpoint_stack = None
    for it in range(first_iter + 1, opt.iterations + 1):
        # progressive SH
        if it % 500 == 0:
            gaussians.oneupSHdegree()

        # random camera selection
        if not viewpoint_stack:
            viewpoint_stack = scene.getCameras().copy()
            random.shuffle(viewpoint_stack)
            if len(viewpoint_stack) > 2000:
                viewpoint_stack = viewpoint_stack[:2000]
        cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack) - 1))
        cam.load2device(args.device)

        # ------------- forward pass ---------------------------------- #
        codedict = {
            "R":      torch.tensor(cam.R, device=args.device),
            "T":      torch.tensor(cam.T, device=args.device),
            "roots":  gaussians.get_roots_xyz[None],
            "theta":  gaussians.theta,
            "beta":   gaussians.beta,
        }

        verts_final, rot_delta, scale_coef = deform_model.decode(gaussians, codedict)
        gaussians.update_xyz_rot_scale(verts_final, rot_delta, scale_coef)

        render_pkg = render(cam, gaussians, ppt, background)
        img_render = render_pkg["render"]

        # ground‑truth composite
        gt_img   = cam.original_image
        alpha    = cam.hair_mask
        orient   = cam.hair_orient             # (H,W,2)
        gt_img   = gt_img * alpha + bg_image * (1 - alpha)

        # ------------------------------------------------------------------ #
        # losses
        # ------------------------------------------------------------------ #
        loss_h  = huber_loss(img_render, gt_img, delta=0.1, reduction="mean")
        loss_o  = orientation_loss(cam, gaussians, alpha, orient)

        strand_pts = verts_final.reshape(
            gaussians.num_strands, STRAND_VERTEX_COUNT + 1, 3)

        loss_len = strand_length_loss(
            strand_pts,
            L_max = opt.max_strand_len,
            delta = opt.delta_strand_len)

        loss_nei = neighbour_orientation_loss(
            strand_pts[:, :STRAND_VERTEX_COUNT, :],
            k = opt.k_neigh)

        lambda_h = getattr(opt, "lambda_huber", 1.0)
        loss     = (lambda_h          * loss_h +
                    opt.lambda_orient * loss_o +
                    opt.lambda_len    * loss_len +
                    opt.lambda_neigh  * loss_nei)

        loss.backward()

        # ------------------------------------------------------------------ #
        # optimiser step
        # ------------------------------------------------------------------ #
        with torch.no_grad():
            if it < opt.iterations:
                gaussians.optimizer.step()
                deform_model.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform_model.optimizer.zero_grad(set_to_none=True)

        # ------------------------------------------------------------------ #
        # logging / visualisation
        # ------------------------------------------------------------------ #
        if it % 20 == 0:
            log_dict = {
                "loss/total":  loss.item(),
                "loss/huber":  loss_h.item(),
                "loss/orient": loss_o.item(),
                "loss/len":    loss_len.item(),
                "loss/neigh":  loss_nei.item(),
                "iter":        it,
            }
            print(f"[{it:06d}] "
                  f"huber {loss_h.item():.4f}  "
                  f"orient {loss_o.item():.4f}  "
                  f"len {loss_len.item():.4f}  "
                  f"neigh {loss_nei.item():.4f}  "
                  f"→ total {loss.item():.4f}")
            if _use_wandb:
                wandb.log(log_dict, step=it)

        if it % 250 == 0 or it == 1:
            canvas = np.zeros((args.image_res, args.image_res * 2, 3), dtype=np.uint8)
            gt_np  = (gt_img.clamp(0,1)*255).permute(1,2,0).cpu().numpy().astype(np.uint8)
            ren_np = (img_render.clamp(0,1)*255).permute(1,2,0).cpu().numpy().astype(np.uint8)
            canvas[:, :args.image_res] = gt_np
            canvas[:, args.image_res:] = ren_np
            cv2.imwrite(os.path.join(train_dir, f"{it:06d}.png"),
                        canvas[:, :, ::-1])        # BGR for OpenCV
            save_tensor_to_ply(verts_final,
                               os.path.join(train_dir, f"{it:06d}.ply"))
            if _use_wandb:
                wandb.log({"preview": wandb.Image(canvas[:, :, ::-1])}, step=it)

        if it % 5000 == 0:
            torch.save(
                (deform_model.capture(), gaussians.capture(), it),
                os.path.join(model_dir, f"chkpnt_{it:06d}.pth"))
            print(f"\n[ITER {it}] Checkpoint saved.\n")

        cam.load2device("cpu")