# visualize_checkpoint_render.py

import os, argparse, torch, imageio, numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from hair.hair_models import Perm
from scene.gaussian_perm import GaussianPerm
from scene import Scene_mica
from src.perm_deform_model import PermDeformModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import to_image_np

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("perm-visualize-render")

    ModelParams(parser)
    OptimizationParams(parser)
    PipelineParams(parser)

    parser.add_argument("--start_checkpoint", type=str, required=True)
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--idname", type=str, required=True)
    parser.add_argument("--image_res", type=int, default=512)
    parser.add_argument("--out_dir", type=str, required=True)

    args = parser.parse_args()
    args.device = "cuda"

    os.makedirs(args.out_dir, exist_ok=True)

    lp = ModelParams.extract(args)
    pp = PipelineParams.extract(args)
    op = OptimizationParams.extract(args)

    # Load static FLAME mask
    scalp_mask = torch.load("/home/alrivero/ENPC/re_perm/flame/FLAME_masks/FLAME_masks.pkl", map_location="cpu")["scalp"]

    data_dir     = os.path.join(args.source_path, args.idname)
    mica_datadir = os.path.join(data_dir, "track_out", args.idname)

    # Initialize models
    perm = Perm(
        lp.perm_path,
        lp.obj_head_path,
        scalp_vertex_idxs=scalp_mask,
        scalp_bounds=[0.1870, 0.8018, 0.4011, 0.8047],
        mesh_scale=100.0
    ).to(args.device)

    pseudo_roots = perm.hair_roots.load_txt(lp.loaded_roots_path)[0]
    start_hair_style = np.load(lp.emp_hair_path) if lp.emp_hair_path else None

    gaussians = GaussianPerm(perm, pseudo_roots, start_hair_style, lp.sh_degree).to(args.device)
    deform_model = PermDeformModel(perm, args.device).to(args.device)

    m_params, g_params, _ = torch.load(args.start_checkpoint, map_location=args.device)
    deform_model.restore(m_params)
    gaussians.restore(g_params, op)
    gaussians.eval()
    deform_model.eval()

    # Background setup
    background = torch.ones(3, args.image_res, args.image_res, device=args.device) if lp.white_background else torch.tensor([0, 1, 0], device=args.device)[:, None, None].expand(3, args.image_res, args.image_res)

    # Setup scene
    scene = Scene_mica(data_dir, mica_datadir, train_type=0, white_background=lp.white_background, device=args.device)
    cameras = scene.getCameras()

    video_render, video_seg, video_depth = [], [], []

    for cam in tqdm(cameras, desc="Rendering"):
        cam.load2device(args.device)

        codedict = {
            "R":     torch.tensor(cam.R, device=args.device),
            "T":     torch.tensor(cam.T, device=args.device),
            "roots": gaussians.get_roots_xyz[None],
            "theta": gaussians.theta,
            "beta":  gaussians.beta,
        }

        verts_final, guide_final, rot_delta, scale_coef = deform_model.decode(gaussians, codedict)
        gaussians.update_xyz_rot_scale(verts_final, rot_delta, scale_coef)

        render_pkg = render(cam, gaussians, pp, background.mean(dim=(1, 2)))

        img_render  = render_pkg["render"]
        img_segment = render_pkg["segment"]
        img_depth   = render_pkg["depth"][0]

        video_render.append(to_image_np(img_render))
        video_seg.append(to_image_np(img_segment))
        depth_vis = img_depth.expand(3, -1, -1) if img_depth.ndim == 2 else img_depth
        video_depth.append(to_image_np(depth_vis))

        cam.load2device("cpu")

    imageio.mimsave(os.path.join(args.out_dir, "render.mp4"), video_render, fps=10)
    imageio.mimsave(os.path.join(args.out_dir, "segmentation.mp4"), video_seg, fps=10)
    imageio.mimsave(os.path.join(args.out_dir, "depth.mp4"), video_depth, fps=10)
    print(f"[✓] Rendered and saved all videos to {args.out_dir}")

if __name__ == "__main__":
    main()