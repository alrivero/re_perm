import os, sys
import random
import json
from PIL import Image
import torch
import pickle
import math
import numpy as np
import cv2
from tqdm import tqdm
from scipy.special import betaln, digamma, polygamma
from pathlib import Path

from scene.gaussian_model import GaussianModel
from scene.gaussian_perm import GaussianPerm
from scene.cameras import Camera
from arguments import ModelParams
from utils.general_utils import PILtoTensor
from utils.graphics_utils import focal2fov, fov2focal
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat


class Scene_mica:
    @staticmethod
    def _fit_beta_spike(
        mask_arr: np.ndarray,
        spike_thresh: float = 0.999,
        max_iter: int = 5,
        eps: float = 1e-8
    ):
        """
        Fit p(x) = pi*delta(x=1) + (1-pi)*Beta(x; alpha_param,beta_param)
        via one EM cycle + Newton on the Beta tail.
        Returns:
          (pi, alpha_param, beta_param, neg_log_likelihood, non_one_frac)
        """

        # 1) flatten & filter zeros
        xs = mask_arr.ravel().astype(np.float64)
        xs = xs[xs > 0.0]
        N = xs.size
        if N < 10:
            return None

        # 2) hard E-step: assign spike vs tail
        is_spike = xs >= spike_thresh
        n_spike  = int(is_spike.sum())
        tail     = xs[~is_spike]
        n_tail   = tail.size
        non_one_frac = n_tail / float(N)
        if n_tail < 5:
            return None

        # 3) M-step for pi
        pi = n_spike / float(N)

        # 4) init alpha/beta by method-of-moments on tail
        mean_tail = tail.mean()
        var_tail  = tail.var()
        common    = mean_tail*(1.0-mean_tail)/(var_tail + eps) - 1.0
        alpha_param = max(mean_tail * common, eps)
        beta_param  = max((1.0-mean_tail) * common, eps)

        # 5) refine alpha/beta by Newton steps
        for _ in range(max_iter):
            # gradient of tail log-likelihood
            grad_a = tail.size * (digamma(alpha_param + beta_param) - digamma(alpha_param)) \
                     + np.log(tail+eps).sum()
            grad_b = tail.size * (digamma(alpha_param + beta_param) - digamma(beta_param)) \
                     + np.log(1-tail+eps).sum()
            # approximate Hessian diagonals
            hess_aa = tail.size * (polygamma(1, alpha_param + beta_param) - polygamma(1, alpha_param))
            hess_bb = tail.size * (polygamma(1, alpha_param + beta_param) - polygamma(1, beta_param))
            # Newton update (damped by adding eps)
            alpha_param = max(alpha_param - grad_a/(hess_aa+eps), eps)
            beta_param  = max(beta_param  - grad_b/(hess_bb+eps), eps)

        # 6) compute complete-data log-likelihood
        ll_spike = n_spike * np.log(pi + eps)
        ll_tail  = n_tail * np.log(1.0 - pi + eps)
        ll_tail += (alpha_param-1.0)*np.log(tail+eps).sum()
        ll_tail += (beta_param-1.0) * np.log(1.0-tail+eps).sum()
        ll_tail -= n_tail * betaln(alpha_param, beta_param)
        log_lik  = ll_spike + ll_tail

        # return negative log-likelihood as “score” to minimize
        return pi, alpha_param, beta_param, -log_lik, non_one_frac

    def __init__(self, datadir, white_background, device, img_dim=(1280, 720), focal_scale=0.5):
        ## train_type: 0 for train, 1 for test, 2 for eval
        images_folder = os.path.join(datadir, "images")
        hair_mask_folder = os.path.join(datadir, "hair_mask")
        hair_orient_folder = os.path.join(datadir, "orient_map")

        cameras_extrinsic_file = os.path.join(datadir, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(datadir, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

        cam_res_path = os.path.join(datadir, "camera_res.pkl")

        # Load camera residuals if the fitted camera file exists
        params_cam_rotation, params_cam_translation, params_cam_fov = {}, {}, {}
        if os.path.exists(cam_res_path):
            print(f"✅ [Scene] Loading fitted camera residuals from:\n  {cam_res_path}")
            try:
                with open(cam_res_path, 'rb') as f:
                    params_cam_rotation, params_cam_translation, params_cam_fov = pickle.load(f)
            except Exception as e:
                print(f"❌ [Scene] Error loading fitted cameras: {e}. Using COLMAP defaults.")
        else:
            print("⚠️ [Scene] No fitted camera file found. Using COLMAP defaults.")

        H = img_dim[0]
        W = img_dim[1]

        self.cameras_extent = 0.547 # 3 times bouding sphere of flame mesh
        self.N_frames = len(os.listdir(images_folder))
        self.bg_image = torch.zeros((3, img_dim[0], img_dim[1]))
        if white_background:
            self.bg_image[:, :, :] = 1
        else:
            self.bg_image[1, :, :] = 1

        self.cameras = []

        self.target_dist  = None          # will hold best mixture
        best_U            = float('inf')  # smallest so far
        best_idx          = -1
        best_params       = None

        for frame_id, key in enumerate(tqdm(cam_extrinsics, desc="Loading camera data")):
            # 1. Gather our COLMAP cameras
            extr = cam_extrinsics[key]
            intr = cam_intrinsics[extr.camera_id]
            image_name_ori = str(int(Path(cam_extrinsics[key].name).stem)).zfill(6)

            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)

            # Apply focal_scale to adjust intrinsics for the target rendering resolution
            if intr.model=="SIMPLE_PINHOLE":
                focal = intr.params[0] * focal_scale
                FovY = focal2fov(focal, H)
                FovX = focal2fov(focal, W)
            elif intr.model=="PINHOLE":
                focal_x = intr.params[0] * focal_scale
                focal_y = intr.params[1] * focal_scale
                FovY = focal2fov(focal_y, H)
                FovX = focal2fov(focal_x, W)
            else:
                assert False, "Colmap camera model not handled."

            # image data
            try:
                image_path = os.path.join(images_folder, image_name_ori+'.png')
                image = Image.open(image_path)
            except:
                continue
            resized_image_rgb = PILtoTensor(image)
            gt_image = resized_image_rgb[:3, ...]

            # hair_mask
            hair_mask_path = os.path.join(hair_mask_folder, image_name_ori+'.png')
            hair_mask = Image.open(hair_mask_path)
            hair_mask = PILtoTensor(hair_mask)
            hair_mask = (hair_mask - hair_mask.min()) / (hair_mask.max() - hair_mask.min() + 1e-8)

            # hairstep map
            hair_orient_path = os.path.join(hair_orient_folder, image_name_ori+'.png')
            hair_orient = Image.open(hair_orient_path)
            hair_orient = PILtoTensor(hair_orient)

            fit = self._fit_beta_spike(hair_mask.cpu().numpy())
            if fit is not None:
                pi, alpha, beta, U, non_one_frac = fit
                if U < best_U:
                    best_U            = U
                    best_idx          = frame_id
                    best_params       = (pi, alpha, beta)
                    best_non_one_frac = non_one_frac

            # Get residuals for the current camera, defaulting to None if not found
            rotation_res = params_cam_rotation.get(image_name_ori)
            translation_res = params_cam_translation.get(image_name_ori)
            fov_res = params_cam_fov.get(image_name_ori)

            camera_indiv = Camera(colmap_id=frame_id, R=R, T=T,
                                FoVx=FovX, FoVy=FovY,
                                image=gt_image, hair_mask=hair_mask, hair_orient=hair_orient,
                                image_name=image_name_ori, uid=frame_id,
                                # Pass the loaded residuals to the camera constructor
                                rotation_res=rotation_res,
                                translation_res=translation_res,
                                fov_res=fov_res)
            self.cameras.append(camera_indiv)

        if best_params is not None:
            pi_best, alpha_best, beta_best = best_params
            self.target_dist = dict(
                frame_idx      = best_idx,
                pi             = pi_best,
                alpha          = alpha_best,
                beta           = beta_best,
                U              = best_U,
                non_one_frac   = non_one_frac,
            )
        else:
            print("[Scene_mica] Warning: no valid hair-mask tail found; "
                  "target_dist left as None.")

        self.cameras = np.array(self.cameras)

    def getCameras(self):
        return self.cameras