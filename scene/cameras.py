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
import torch.nn.functional as F
import numpy as np
# Assuming these utility functions are available in your project's utils path
from utils.graphics_utils import getWorld2View, getWorld2View2, getProjectionMatrix

# Helper function for 6D rotation representation
def ortho2rotation(poses):
    """
    Converts a 6D representation into a rotation matrix.
    """
    def proj_u2a(u, a):
        inner_prod = (u * a).sum(-1, keepdim=True)
        norm2 = torch.clamp((u ** 2).sum(-1, keepdim=True), min=1e-8)
        factor = inner_prod / (norm2 + 1e-10)
        return factor * u

    x_raw = poses[..., 0:3]
    y_raw = poses[..., 3:6]

    x = F.normalize(x_raw, dim=-1)
    y = F.normalize(y_raw - proj_u2a(x, y_raw), dim=-1)
    z = torch.cross(x, y, dim=-1)

    return torch.stack([x, y, z], dim=-1)

# Helper function for 3D axis-angle rotation representation
def axis_angle_to_matrix(axis_angle):
    """
    Convert an axis-angle representation to a rotation matrix using Rodrigues' formula.
    """
    angle = torch.norm(axis_angle, p=2)
    if angle < 1e-8: # If angle is very small, return identity matrix
        return torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    
    axis = axis_angle / angle
    K = torch.zeros((3, 3), device=axis_angle.device, dtype=axis_angle.dtype)
    K[0, 1], K[0, 2] = -axis[2], axis[1]
    K[1, 0], K[1, 2] = axis[2], -axis[0]
    K[2, 0], K[2, 1] = -axis[1], axis[0]
    
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    
    # Rodrigues' rotation formula
    return torch.eye(3, device=axis.device, dtype=axis.dtype) + sin_a * K + (1 - cos_a) * torch.matmul(K, K)


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, image_name, uid,
                 hair_mask=None, body_mask=None, hair_orient=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0,
                 rotation_res=None, translation_res=None, fov_res=None):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self._FoVx = torch.tensor(FoVx, dtype=torch.float32)
        self._FoVy = torch.tensor(FoVy, dtype=torch.float32)
        self.image_name = image_name
        self.trans = trans
        self.scale = scale
        
        self._rotation_res = rotation_res.data
        self._translation_res = translation_res.data
        self._fov_res = fov_res.data

        self.device = "cpu"

        self.original_image = image.clamp(0.0, 1.0)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.hair_mask = hair_mask
        self.body_mask = body_mask
        self.hair_orient = hair_orient
        
        self.zfar = 100.0
        self.znear = 0.01

        # Store the BASE 4x4 world-to-camera matrix. The property will handle residuals.
        self._base_w2c = torch.tensor(getWorld2View(R, T), dtype=torch.float32)
        
        # Base (pre-residual) 4x4 camera parameters for other dynamic properties
        self._colmap_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale), dtype=torch.float32)
        self._base_world_view_transform = self._colmap_transform.transpose(0, 1)
        self._base_projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self._FoVx, fovY=self._FoVy).transpose(0, 1)
        self._base_full_proj_transform = (self._base_world_view_transform.unsqueeze(0) @ self._base_projection_matrix.unsqueeze(0)).squeeze(0)
        self._base_camera_center = self._base_world_view_transform.inverse()[3, :3]

        tan_fovx = np.tan(self._FoVx.item() / 2.0)
        tan_fovy = np.tan(self._FoVy.item() / 2.0)
        self.focal_x = self.image_width / (2.0 * tan_fovx)
        self.focal_y = self.image_height / (2.0 * tan_fovy)

    @property
    def FoVx(self):
        if self._fov_res is not None:
            return self._FoVx + self._fov_res[0]
        return self._FoVx

    @property
    def FoVy(self):
        if self._fov_res is not None:
            return self._FoVy + self._fov_res[1]
        return self._FoVy

    @property
    def _residual_transform(self):
        """Computes the 4x4 residual transformation matrix from the residual tensors."""
        if self._rotation_res.shape[0] == 3:
            R_a = axis_angle_to_matrix(self._rotation_res)
        elif self._rotation_res.shape[0] == 6:
            R_a = ortho2rotation(self._rotation_res)
        else:
            raise ValueError(f"Unsupported rotation residual size: {self._rotation_res.shape}")

        t_a = self._translation_res
        residual_transform = torch.eye(4, device=self.device, dtype=torch.float32)
        residual_transform[:3, :3] = R_a
        residual_transform[:3, 3] = t_a
        return residual_transform

    @property
    def w2c(self):
        """
        Returns the 3x4 world-to-camera matrix, adjusted by residuals if available.
        """
        if self._rotation_res is not None and self._translation_res is not None:
            # Apply the residual transformation to the base 4x4 w2c matrix
            refined_w2c_4x4 = self._residual_transform @ self._base_w2c
            # Return the top 3 rows (the 3x4 matrix)
            return refined_w2c_4x4[:3, :]
        else:
            # Return the top 3 rows of the base 4x4 matrix
            return self._base_w2c[:3, :]

    @property
    def world_view_transform(self):
        if self._rotation_res is not None and self._translation_res is not None:
            # Note: This transform includes the scene normalization (trans, scale)
            return (self._residual_transform @ self._colmap_transform).transpose(0, 1)
        return self._base_world_view_transform

    @property
    def projection_matrix(self):
        if self._fov_res is not None:
            return getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).to(self.device).transpose(0, 1)
        return self._base_projection_matrix

    @property
    def full_proj_transform(self):
        return (self.world_view_transform.unsqueeze(0) @ self.projection_matrix.unsqueeze(0)).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    def load2device(self, device="cuda"):
        try:
            target_device = torch.device(device)
        except Exception as e:
            print(f"[Warning] Custom device {device} failed: {e}. Falling back to cuda.")
            target_device = torch.device("cuda")

        self.original_image = self.original_image.to(target_device)
        if self.hair_mask is not None: self.hair_mask = self.hair_mask.to(target_device)
        if self.body_mask is not None: self.body_mask = self.body_mask.to(target_device)
        if self.hair_orient is not None: self.hair_orient = self.hair_orient.to(target_device)

        self._base_w2c = self._base_w2c.to(target_device)

        self._FoVx = self._FoVx.to(target_device)
        self._FoVy = self._FoVy.to(target_device)
        
        self._colmap_transform = self._colmap_transform.to(target_device)
        self._base_world_view_transform = self._base_world_view_transform.to(target_device)
        self._base_projection_matrix = self._base_projection_matrix.to(target_device)
        self._base_full_proj_transform = self._base_full_proj_transform.to(target_device)
        self._base_camera_center = self._base_camera_center.to(target_device)

        if self._rotation_res is not None: self._rotation_res = self._rotation_res.to(target_device)
        if self._translation_res is not None: self._translation_res = self._translation_res.to(target_device)
        if self._fov_res is not None: self._fov_res = self._fov_res.to(target_device)
        
        self.device = target_device

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