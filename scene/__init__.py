import os, sys
import random
import json
from PIL import Image
import torch
import math
import numpy as np
import cv2
from tqdm import tqdm

from scene.gaussian_model import GaussianModel
from scene.gaussian_perm import GaussianPerm
from scene.cameras import Camera
from arguments import ModelParams
from utils.general_utils import PILtoTensor
from utils.graphics_utils import focal2fov, fov2focal


class Scene_mica:
    def __init__(self, datadir, mica_datadir, train_type, white_background, device):
        ## train_type: 0 for train, 1 for test, 2 for eval
        frame_delta = 1 # default mica-tracking starts from the second frame
        images_folder = os.path.join(datadir, "imgs")
        parsing_folder = os.path.join(datadir, "parsing")
        alpha_folder = os.path.join(datadir, "alpha")
        hair_mask_folder = os.path.join(datadir, "hair_mask")
        hair_orient_folder = os.path.join(datadir, "hairstep")
        flame_params_folder = os.path.join(datadir, "flame_params")

        contents = json.load(open(os.path.join(datadir, "cam", "transforms.json"), "r"))

        self.cameras_extent = 0.547 # 3 times bouding sphere of flame mesh
        self.N_frames = len(os.listdir(flame_params_folder))
        self.bg_image = torch.zeros((3, 720, 720))
        if white_background:
            self.bg_image[:, :, :] = 1
        else:
            self.bg_image[1, :, :] = 1

        self.cameras = []

        test_num = 500
        eval_num = 50
        max_train_num = 10000
        train_num = min(max_train_num, self.N_frames - test_num)


        fovx = contents["camera_angle_x"]
        frames = contents["frames"]

        if train_type == 0:
            range_down = 0
            range_up = train_num
        if train_type == 1:
            range_down = self.N_frames - test_num
            range_up = self.N_frames
        if train_type == 2:
            range_down = self.N_frames - eval_num
            range_up = self.N_frames

        for frame_id in tqdm(range(0, 1879)):
            image_name_ori = str(frame_id).zfill(5)

            image_path = os.path.join(images_folder, image_name_ori+'.png')
            image = Image.open(image_path)
            resized_image_rgb = PILtoTensor(image)
            gt_image = resized_image_rgb[:3, ...]

            FovX = frames[frame_id]["camera_angle_x"]
            FovY = frames[frame_id]["camera_angle_y"]

            flame_path = os.path.join(flame_params_folder, image_name_ori+'.npz')
            flame_params = np.load(flame_path)

            exp_param = torch.as_tensor(flame_params["expr"])
            shape_param = torch.as_tensor(flame_params["shape"])[None]
            eyes_pose = torch.as_tensor(flame_params["eyes_pose"])
            jaw_pose = torch.as_tensor(flame_params["jaw_pose"])
            neck_pose = torch.as_tensor(flame_params["neck_pose"])

            rot_raw = np.array(flame_params["rotation"])
            trans_raw = np.array(flame_params["translation"])

            rot_vec = rot_raw.reshape(-1)
            R_w2c_f = cv2.Rodrigues(rot_vec)[0]                   # world→cam  (FLAME axes)
            t_w2c_f = trans_raw.reshape(3, 1)                     # translation

            # 1. pivot-correct translation so rotation is about head root joint
            p_root  = np.array([[-0.0009, -0.1400, -0.0841]]).T   # (3,1)
            t_w2c_f = t_w2c_f - R_w2c_f @ p_root + p_root

            # 2. add tracker’s fixed camera offset (-1 in Z_cam, FLAME axes)
            t0_f    = np.array([[0.0, 0.0, -1.0]]).T
            t_w2c_f = t_w2c_f + t0_f

            # 3. flip axes to GS (Y↓, left-handed)
            S_cam   = np.diag([ 1, -1, -1])   # flip Y & Z for basis vectors
            R_w2c_gs = S_cam @ R_w2c_f
            t_w2c_gs = (S_cam @ t_w2c_f).reshape(3)

            # 4. Camera() expects camera→world, row-major
            R = R_w2c_gs.T.astype(np.float32)   # (3,3)
            T = t_w2c_gs.astype(np.float32)     # (3,)

            # alpha
            alpha_path = os.path.join(alpha_folder, image_name_ori+'.png')
            alpha = Image.open(alpha_path)
            alpha = PILtoTensor(alpha)
            alpha = (alpha - alpha.min()) / (alpha.max() - alpha.min() + 1e-8)

            # # if add head mask
            head_mask_path = os.path.join(parsing_folder, image_name_ori+'_neckhead.png')
            head_mask = Image.open(head_mask_path)
            head_mask = PILtoTensor(head_mask)
            # gt_image = gt_image * alpha + self.bg_image * (1 - alpha)
            # gt_image = gt_image * head_mask + self.bg_image * (1 - head_mask)

            # mouth mask
            mouth_mask_path = os.path.join(parsing_folder, image_name_ori+'_mouth.png')
            mouth_mask = Image.open(mouth_mask_path)
            mouth_mask = PILtoTensor(mouth_mask)

            # hair_mask
            hair_mask_path = os.path.join(hair_mask_folder, image_name_ori+'.png')
            hair_mask = Image.open(hair_mask_path)
            hair_mask = PILtoTensor(hair_mask)
            hair_mask = (hair_mask - hair_mask.min()) / (hair_mask.max() - hair_mask.min() + 1e-8)
            hair_mask[hair_mask < 0.99] = 0.0

            # hairstep map
            hair_orient_path = os.path.join(hair_orient_folder, image_name_ori+'.png')
            hair_orient = Image.open(hair_orient_path)
            hair_orient = PILtoTensor(hair_orient)

            # depth map
            depth_path = os.path.join(datadir, "depth", image_name_ori + ".npy")
            depth_map_np = np.load(depth_path)                      # (H, W) float
            depth_map = torch.from_numpy(depth_map_np)

            camera_indiv = Camera(colmap_id=frame_id, R=R, T=T, 
                                FoVx=FovX, FoVy=FovY, 
                                image=gt_image, head_mask=head_mask, mouth_mask=mouth_mask, hair_mask=hair_mask, hair_orient=hair_orient, depth_map=depth_map,
                                exp_param=exp_param, shape_param=shape_param, eyes_pose=eyes_pose, jaw_pose=jaw_pose, neck_pose=neck_pose,
                                image_name=image_name_ori, uid=frame_id, data_device=device)
            self.cameras.append(camera_indiv)
        
        self.cameras = np.array(self.cameras)
    
    def getCameras(self):
        return self.cameras





    
