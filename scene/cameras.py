#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getWorld2View

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, head_mask, mouth_mask, hair_mask, hair_orient,
                 exp_param, eyes_pose, eyelids, jaw_pose,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.head_mask = head_mask
        self.mouth_mask = mouth_mask
        self.hair_mask = hair_mask
        self.hair_orient = hair_orient
        self.exp_param = exp_param
        self.eyes_pose = eyes_pose
        self.eyelids = eyelids
        self.jaw_pose = jaw_pose

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.w2c = torch.tensor(getWorld2View(R, T))
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def load2device(self, data_device="cuda"):
        self.original_image = self.original_image.clamp(0.0, 1.0).to(data_device)
        self.head_mask = self.head_mask.to(data_device)
        self.mouth_mask = self.mouth_mask.to(data_device)
        self.hair_mask = self.hair_mask.to(data_device)
        self.hair_orient = self.hair_orient.to(data_device)
        self.exp_param = self.exp_param.to(data_device)
        self.eyes_pose = self.eyes_pose.to(data_device)
        self.eyelids = self.eyelids.to(data_device)
        self.jaw_pose = self.jaw_pose.to(data_device)

        self.w2c = self.w2c.to(data_device)

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

