import os, sys 
import random
import numpy as np
import torch
import torch.nn as nn
import argparse
import cv2
import lpips
import pickle
import debug

from hair import save_hair
from hair.hair_models import Perm
from scene.gaussian_perm import GaussianPerm

from scene import Scene_mica
from src.perm_deform_model import PermDeformModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.loss_utils import huber_loss
from utils.general_utils import normalize_for_percep, save_tensor_to_ply


def set_random_seed(seed):
    r"""Set random seeds for everything.

    Args:
        seed (int): Random seed.
        by_rank (bool):
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--idname', type=str, default='id1_25', help='id name')
    parser.add_argument('--image_res', type=int, default=512, help='image resolution')
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.device = "cuda"
    lpt = lp.extract(args)
    opt = op.extract(args)
    ppt = pp.extract(args)

    torch.autograd.set_detect_anomaly(True)

    batch_size = 1
    set_random_seed(args.seed)

    percep_module = lpips.LPIPS(net='vgg').to(args.device)


    ## dataloader
    data_dir = os.path.join(args.source_path, args.idname)
    mica_datadir = os.path.join(args.source_path, args.idname, 'track_out', args.idname)
    log_dir = os.path.join(data_dir, 'log')
    train_dir = os.path.join(log_dir, 'train')
    model_dir = os.path.join(log_dir, 'ckpt')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    first_iter = 0
    
    scalp_mask = pickle.load(open(
        "/home/alrivero/ENPC/re_perm/flame/FLAME_masks/FLAME_masks.pkl", "rb"),
        encoding="latin1")["scalp"]

    perm = Perm(
        lpt.perm_path,
        lpt.obj_head_path,
        scalp_vertex_idxs=scalp_mask,
        scalp_bounds=[0.1870, 0.8018, 0.4011, 0.8047],
        mesh_scale=100.0
    ).to(args.device)

    pseudo_roots = perm.hair_roots.load_txt(lpt.loaded_roots_path)[0]
    # ps_x = pseudo_roots[:, [0]]
    # ps_y = pseudo_roots[:, [1]]
    # ps_z = pseudo_roots[:, [2]]
    # pseudo_roots = torch.cat([ps_x, -ps_z, ps_y], dim=-1)

    start_hair_style = None
    if lpt.emp_hair_path:
        start_hair_style = np.load(lpt.emp_hair_path)
    else:
        start_hair_style = None

    gaussians = GaussianPerm(perm, pseudo_roots, start_hair_style, lpt.sh_degree).to(args.device)
    gaussians.roots = gaussians.roots.to(args.device)
    gaussians.training_setup(opt)

    ## deform model
    DeformModel = PermDeformModel(perm, args.device).to(args.device)
    DeformModel.training_setup()

    if args.start_checkpoint:
        (model_params, gauss_params, first_iter) = torch.load(args.start_checkpoint)
        DeformModel.restore(model_params)
        gaussians.restore(gauss_params, opt)

    bg_color = [1, 1, 1] if lpt.white_background else [0, 1, 0]
    bg_image = torch.zeros((3, 512, 512)).to(args.device)
    if lpt.white_background:
        bg_image[:, :, :] = 1
    else:
        bg_image[1, :, :] = 1
    background = torch.tensor(bg_color, dtype=torch.float32, device=args.device)
    
    scene = Scene_mica(data_dir, mica_datadir, train_type=0, white_background=lpt.white_background, device = args.device)

    codedict = {}
    viewpoint_stack = None
    first_iter += 1
    mid_num = 15000

    for iteration in range(first_iter, opt.iterations + 1):
        # Every 500 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getCameras().copy()
            random.shuffle(viewpoint_stack)
            if len(viewpoint_stack)>2000:
                viewpoint_stack = viewpoint_stack[:2000]
        viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack)-1))
        viewpoint_cam.load2device(args.device)

        frame_id = viewpoint_cam.uid

        # For now, assuming camera/subject movement is the same
        codedict['R'] = torch.tensor(viewpoint_cam.R).to(args.device)
        codedict['T'] = torch.tensor(viewpoint_cam.T).to(args.device)
        codedict["roots"] = gaussians.get_roots_xyz[None]
        codedict["theta"] = gaussians.theta
        codedict["beta"] = gaussians.beta

        # import pdb; pdb.set_trace()

        verts_final, rot_delta, scale_coef = DeformModel.decode(gaussians, codedict)
        gaussians.update_xyz_rot_scale(verts_final, rot_delta, scale_coef)

        # Render
        # import pdb; pdb.set_trace()
        render_pkg = render(viewpoint_cam, gaussians, ppt, background)
        image = render_pkg["render"]

        # Composite ground-truth against background
        gt_image = viewpoint_cam.original_image
        alpha    = viewpoint_cam.hair_mask     # (H,W) mask
        gt_image = gt_image * alpha + bg_image * (1 - alpha)

        # --- Masked Huber Loss using our custom huber_loss ---
        # 1) element-wise Huber with no reduction
        loss_map = huber_loss(image, gt_image, delta=0.1, reduction='none')  # (C,H,W) or (H,W)

        # 2) broadcast mask over channels if needed
        if loss_map.dim() == 3:                  # (C,H,W)
            mask = alpha.unsqueeze(0)            # (1,H,W) -> (C,H,W)
        else:                                    # (H,W)
            mask = alpha

        # 3) zero out non-hair pixels, then average over mask
        masked_loss = loss_map * mask
        loss_huber  = masked_loss.sum() / (mask.sum() + 1e-6)

        # Total loss
        loss = loss_huber  # + other terms…

        loss.backward()

        with torch.no_grad():
            # Optimizer step
            if iteration < opt.iterations :
                gaussians.optimizer.step()
                DeformModel.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                DeformModel.optimizer.zero_grad(set_to_none = True)
            
            # print loss
            if iteration % 20 == 0:
                if iteration<=mid_num:
                    print("step: %d, huber: %.5f" %(iteration, loss_huber.item()))
                else:
                    print("step: %d, huber: %.5f" %(iteration, loss_huber.item()))
            
            # visualize results
            if iteration % 250 == 0 or iteration==1:
                save_image = np.zeros((args.image_res, args.image_res*2, 3))
                gt_image_np = (gt_image*255.).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
                image = image.clamp(0, 1)
                image_np = (image*255.).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
                save_image[:, :args.image_res, :] = gt_image_np
                save_image[:, args.image_res:, :] = image_np
                cv2.imwrite(os.path.join(train_dir, f"{iteration}.png"), save_image[:,:,[2,1,0]])
                save_tensor_to_ply(verts_final, os.path.join(train_dir, f"{iteration}.ply"))
                
            
            # save checkpoint
            if iteration % 5000 == 0:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((DeformModel.capture(), gaussians.capture(), iteration), model_dir + "/chkpnt" + str(iteration) + ".pth")
        
        viewpoint_cam.load2device("cpu")

           