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
import torch.nn as nn
import sys
from datetime import datetime
import numpy as np
import random
import rff
import trimesh
import pickle
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import rasterize_meshes
from pxr import Usd, UsdGeom, Gf
import torch.nn.functional as F

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer
)

import torch
from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, MeshRasterizer
from pytorch3d.structures import Meshes

def get_linear_noise_func(
        lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = lr_init * (1 - t) + lr_final * t
        return delay_rate * log_lerp

    return helper

def make_p3d_camera(cam, device):
    """
    Build a PyTorch3D PerspectiveCameras from your MiniCam/Camera,
    using focal_x, focal_y (pixels) and image center as principal point.
    """
    H, W = int(cam.image_height), int(cam.image_width)
    # Extrinsics: world->camera
    C2W   = cam.world_view_transform.to(device).float()   # camera→world (4×4)
    R_c2w = C2W[:3, :3]                                   # (3×3)
    t_c2w = C2W[3, :3]                                    # (3,)
    R     = R_c2w.unsqueeze(0)             # (1,3,3)
    T     = t_c2w.unsqueeze(0)   # (1,3)

    # Intrinsics: focal lengths and principal point
    fx, fy = cam.focal_x, cam.focal_y
    px, py = W * 0.5, H * 0.5

    return PerspectiveCameras(
        focal_length    = ((fx, fy),),       # batch of one
        principal_point = ((px, py),),
        R               = R,
        T               = T,
        in_ndc          = False,             # pixel units
        image_size      = [[H, W]],
        device          = device
    )

def compute_occlusion_mask(pc, cam, verts, faces, eps=1e-3, scale=0.96):
    device = pc.get_xyz.device
    if scale != 1.0:
        center = verts.mean(dim=1, keepdim=True)  # (1, V, 3) if batched
        verts = (verts - center) * scale + center
    H, W   = int(cam.image_height), int(cam.image_width)

    # Build the camera
    cameras = make_p3d_camera(cam, device)

    # 1) Rasterize mesh → camera‐space depth_map
    mesh     = Meshes(verts=verts, faces=faces)
    fragments= MeshRasterizer(
                  cameras=cameras,
                  raster_settings=RasterizationSettings(
                      image_size=(H, W),
                      blur_radius=0.0,
                      faces_per_pixel=1
                  )
              )(mesh)
    depth_map = fragments.zbuf[0, ..., 0]
    depth_map = depth_map.flip(0).flip(1)

    # 2) Compute camera‐space Z for each Gaussian
    world_to_view = cameras.get_world_to_view_transform()
    pts_cam_h     = world_to_view.transform_points(pc.get_xyz.unsqueeze(0))  # (1, M, 3)
    z_cam         = pts_cam_h[0, :, 2]                                        # (M,)

    # 3) Project only for pixel coords (u, v)
    screen_pts    = cameras.transform_points_screen(
                        pc.get_xyz.unsqueeze(0),
                        image_size=(H, W)
                    )[0]
    u, v, _       = screen_pts.unbind(-1)
    u_pix         = u.round().long().clamp(0, W-1)
    v_pix         = v.round().long().clamp(0, H-1)
    u_pix = (W - 1) - u_pix
    v_pix = (H - 1) - v_pix

    # 4) Occlusion: same camera‐space depth domain
    in_front     = z_cam > eps
    depth_at_pix = depth_map[v_pix, u_pix]            # (M,)
    back_at_pix   = depth_at_pix == -1
    visible      = in_front & ((z_cam <= depth_at_pix + eps) | back_at_pix)

    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    return visible, depth_map

def average_opacity_per_strand(gaussians):
    """
    Compute the average opacity of all Gaussians grouped by strand.

    Returns:
      avg_op: Tensor of shape (S,) where S = gaussians.num_strands,
              and avg_op[i] is the mean opacity of strand i.
    """
    # 1) fetch per-Gaussian opacities: (M,)
    #    you can swap get_opacity_with_3D_filter for get_opacity if you prefer
    opac = gaussians.get_opacity_with_3D_filter.squeeze(-1)  

    # 2) strand IDs for each Gaussian: (M,)
    strand_id = gaussians._strand_id  

    # 3) prepare accumulator & counts
    S = gaussians.num_strands
    device = opac.device
    sum_op = torch.zeros(S, device=device)
    count = torch.zeros(S, device=device)

    # 4) accumulate sums and counts
    sum_op  = sum_op.index_add_(0, strand_id, opac)
    count   = count.index_add_(0, strand_id, torch.ones_like(opac))

    # 5) avoid division by zero, then compute mean
    count = count.clamp(min=1.0)
    avg_op = sum_op / count

    return avg_op


def average_opacity_for_strand(gaussians, strand_idx):
    """
    Compute the average opacity of all Gaussians attached to strand `strand_idx`.
    """
    opac       = gaussians.get_opacity_with_3D_filter.squeeze(-1)
    strand_id  = gaussians._strand_id
    mask       = (strand_id == strand_idx)

    if mask.sum() == 0:
        raise ValueError(f"No Gaussians found for strand {strand_idx}")
    return opac[mask].mean()

def compute_pixel_footprint(K, depth):
    """
    Given camera intrinsics K (3x3) and a per-pixel depth map (H x W),
    compute the isotropic blur sigma each Gaussian should be
    pre-blurred by to match its screen-space footprint.
    """
    fx = K[0, 0]
    fy = K[1, 1]

    # world‐space blur needed to cover one pixel ∼ depth / focal_length
    sigma_x = depth / fx    # (H,W)
    sigma_y = depth / fy    # (H,W)

    # average to get one isotropic sigma per pixel
    sigma = 0.5 * (sigma_x + sigma_y)
    return sigma

def to_image_np(t: torch.Tensor) -> np.ndarray:
    """
    Convert a (C,H,W) or (B,C,H,W) torch tensor in [0,1] to a uint8 H×W×3 array.
    """
    t = t.detach().clamp(0, 1)
    if t.dim() == 4:                # (B,C,H,W) → take first
        t = t[0]
    if t.dim() == 3:                # (C,H,W)
        arr = t.permute(1, 2, 0).cpu().numpy()          # → (H,W,C)
    elif t.dim() == 2:              # (H,W) mono depth / mask
        g = t.cpu().numpy()
        arr = np.stack([g, g, g], axis=-1)
    else:
        raise RuntimeError(f"Unsupported tensor shape {tuple(t.shape)}")
    return (arr * 255.0).astype(np.uint8)

def export_strands_to_usd(filename, strand_vertices, gaussians):
    S, V, _ = strand_vertices.shape

    # ---------- per-vertex width from Gaussian scales -----------------
    scales       = gaussians.get_scaling.detach()        # (M,3)
    sigma_perp   = scales[:, 0] * 100                          # use x-axis σ
    seg_radius   = 2.0 * sigma_perp                      # diameter
    seg_radius   = seg_radius.view(S, V-1)               # (S, V-1)
    width_tensor = torch.cat([seg_radius[:, :1], seg_radius], dim=1)
    widths_np    = width_tensor.cpu().numpy()            # (S, V)

    # ---------- write USD --------------------------------------------
    stage = Usd.Stage.CreateNew(filename)
    UsdGeom.Scope.Define(stage, "/hair")                 # container prim

    for sid in range(S):
        pts_np  = strand_vertices[sid].detach().cpu().numpy() * 100
        pts     = [Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in pts_np]
        w       = [float(r) for r in widths_np[sid]]

        curve   = UsdGeom.BasisCurves.Define(stage, f"/hair/strand_{sid}")
        curve.CreatePointsAttr(pts)
        curve.CreateWidthsAttr(w)
        curve.CreateCurveVertexCountsAttr([V])
        curve.CreateTypeAttr("linear")
        curve.CreateWrapAttr("nonperiodic")

    stage.Save()
    print(f"Saved {S} strands with variable widths → {filename}")

def export_strands_as_obj(strands: torch.Tensor, path: str):
    """
    strands: (S, V, 3) float32 tensor of your strand vertices
    """
    S, V, _ = strands.shape
    verts = strands.reshape(-1,3).cpu().numpy()
    with open(path, 'w') as f:
        f.write("# hair strands as polylines\n")
        # write all vertices
        for x,y,z in verts:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        # now write a polyline per strand
        idx = 1
        for s in range(S):
            f.write(f"g strand_{s:04d}\n")           # new Blender object/group
            indices = [str(i) for i in range(idx, idx+V)]
            f.write("l " + " ".join(indices) + "\n")
            idx += V

def convert_normal_to_camera_space(
        normals:        torch.Tensor,   # (N,3) world-space normals
        extrinsic_rot:  torch.Tensor,   # (3,3) world→cam rotation  R_wc
        intrinsics:     torch.Tensor,   # (3,3) K
        eps: float = 1e-6):
    """
    Returns 2-D unit directions (+X→right, +Y→up) for each world-space normal,
    safely handling zeros or negatives in the depth (z) channel.
    """
    # 1) rotate into camera coords
    n_cam = normals @ extrinsic_rot.T           # (N,3)

    # 2) extract z
    z = n_cam[:, 2:3]                           # (N,1)

    # 3) compute raw slopes; allow true negatives (they’ll get zeroed soon)
    slopes = n_cam[:, :2] / (z + eps)           # (N,2)  small eps only to avoid div-by-zero

    # 4) apply intrinsics *sans* principal-point
    #    (we only care about direction, so we skip c_x/c_y entirely)
    fx, fy = intrinsics[0,0], intrinsics[1,1]
    slopes = slopes * torch.tensor([fx, fy], device=slopes.device)  # pixel‐scale slopes

    # 5) flip v if +v is down in your convention
    slopes[:,1] = -slopes[:,1]

    return slopes

def quaternion_to_rotation_matrix(quaternions):
    # Ensure quaternions tensor has the correct shape (n x 4)
    assert quaternions.shape[-1] == 4, "Input tensor must be of shape (n, 4)"

    # Unpack quaternion components
    a, b, c, d = quaternions.unbind(dim=-1)

    # Calculate rotation matrix elements
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    ab, ac, ad, bc, bd, cd = a*b, a*c, a*d, b*c, b*d, c*d
 
    # Form the rotation matrix
    rotation_matrix = torch.stack([
        torch.stack([aa + bb - cc - dd, 2 * (bc - ad), 2 * (bd + ac)]),
        torch.stack([2 * (bc + ad), aa - bb + cc - dd, 2 * (cd - ab)]),
        torch.stack([2 * (bd - ac), 2 * (cd + ab), aa - bb - cc + dd])
    ], dim=-2).permute(2, 1, 0)

    return rotation_matrix

def project_gaussians_to_screen_uv(gaussians, camera, mask_to_project):
    """
    Projects a SUBSET of 3D world-space Gaussian centers to integer pixel coordinates.
    """
    # Only project the Gaussians specified by the mask
    xyz = gaussians.get_xyz[mask_to_project]

    if xyz.shape[0] == 0:
        return torch.tensor([], device=xyz.device, dtype=torch.long), \
               torch.tensor([], device=xyz.device, dtype=torch.long)

    # Add homogeneous coordinate
    points = torch.cat((xyz, torch.ones((xyz.shape[0], 1), device=xyz.device)), dim=-1)
    points = points.unsqueeze(0)

    # Get camera matrices
    w2c = camera.w2c.unsqueeze(0)
    k = camera.projection_matrix[:3, :3].unsqueeze(0)
    height, width = camera.image_height, camera.image_width

    # Transform to camera space and perform perspective divide
    points_cam = torch.bmm(points, w2c.transpose(1, 2))
    points_cam = points_cam[:, :, :3]
    points_cam /= (points_cam[:, :, [2]] + 1e-9)

    # Transform to screen space UV coordinates
    points_screen = torch.bmm(points_cam, k.transpose(1, 2))[:, :, :2]
    
    u = (points_screen[:, :, 0] + 1.0) * 0.5 * width
    v = (points_screen[:, :, 1] + 1.0) * 0.5 * height
    
    return u.squeeze(0), v.squeeze(0)


def create_outside_mask(gaussians, camera, occ_mask, gt_alpha):
    """
    Creates a boolean mask identifying Gaussians that project outside the gt_alpha mask.
    This version correctly uses the occlusion mask for efficiency.
    """
    dev = gaussians.get_xyz.device
    
    # --- FIX: Project ONLY the potentially visible Gaussians ---
    visible_indices = occ_mask.nonzero(as_tuple=False).squeeze(1)
    if visible_indices.numel() == 0:
        return torch.zeros(gaussians.num_gaussians, dtype=torch.bool, device=dev)
        
    u, v = project_gaussians_to_screen_uv(gaussians, camera, occ_mask)
    
    # Clamp coordinates to be within the image bounds for sampling
    H, W = gt_alpha.shape[-2:]
    u_clamped = torch.clamp(u.round().long(), 0, W - 1)
    v_clamped = torch.clamp(v.round().long(), 0, H - 1)

    # Sample the mask at the visible Gaussian locations
    mask_values = gt_alpha.squeeze()[v_clamped, u_clamped]
    is_outside_visible = mask_values < 0.1
    
    # Create the final mask for ALL Gaussians
    outside_mask = torch.zeros(gaussians.num_gaussians, dtype=torch.bool, device=dev)
    # Scatter the results back to their original positions
    outside_mask[visible_indices] = is_outside_visible
    
    return outside_mask

def project_to_screen(points, all_k, all_w2c, height=512, width=512):
    points = torch.cat((points, torch.ones((len(points), 1)).cuda()), dim=-1)
    points = points.unsqueeze(0).expand(len(all_k), -1, -1)

    # Now convert into camera space and project onto focal plane
    points_cam = torch.bmm(points, all_w2c.transpose(1, 2))
    points_cam = points_cam[:, :, :3]
    points_cam /= points_cam[:, :, [2]]

    # Now bring them into screen space and define uvs
    points_screen = torch.bmm(points_cam, all_k.transpose(1, 2))[:, :, :2]
    points_screen[:, :, 0] = (points_screen[:, :, 0] + 1.0) * 0.5 * width
    points_screen[:, :, 1] = (points_screen[:, :, 1] + 1.0) * 0.5 * height
    uvs = torch.round(points_screen).long()

    return uvs

def save_tensor_to_ply(points: torch.Tensor, path: str):
    """
    Save a (N,3) torch tensor as a PLY point cloud (vertex only, float).
    """
    points = points.detach().cpu().numpy()
    N = points.shape[0]

    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )

    with open(path, 'w') as f:
        f.write(header)
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

def save_tensor_to_obj(points: torch.Tensor, filename: str):
    """
    Save an (N,3) torch.Tensor point cloud to an OBJ file via trimesh.

    Args:
        points:   torch.Tensor of shape (N,3)
        filename: Path to the output .obj file (should end in .obj)
    """
    # 1) detach and move to CPU, convert to (N,3) numpy
    pts = points.detach().cpu().numpy()
    # 2) build a PointCloud and export by extension
    pc = trimesh.PointCloud(pts)
    pc.export(filename)  # trimesh infers OBJ from .obj extension

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
    
def PILtoTensor(pil_image):
    resized_image = torch.from_numpy(np.array(pil_image)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))




def quatProduct_batch(q1, q2):
    r1 = q1[:,0] # [B]
    r2 = q2[:,0]
    v1 = torch.stack((q1[:,1], q1[:,2], q1[:,3]), dim=-1) #[B,3]
    v2 = torch.stack((q2[:,1], q2[:,2], q2[:,3]), dim=-1)

    r = r1 * r2 - torch.sum(v1*v2, dim=1) # [B]
    v = r1.unsqueeze(1) * v2 + r2.unsqueeze(1) * v1 + torch.cross(v1, v2) #[B,3]
    q = torch.stack((r, v[:,0], v[:,1], v[:,2]), dim=1)

    return q

def load_binary_pickle(filepath):
    with open(filepath, 'rb') as f:
        if sys.version_info >= (3, 0):
            data = pickle.load(f, encoding='latin1')
        else:
            data = pickle.load(f)
    return data

def a_in_b_torch(a, b):
    ainb = torch.isin(a, b)
    return ainb

def normalize_for_percep(input, mod_n = 64):
    h, w = input.shape[1:3]
    # delta_h = ((h-1)//mod_n + 1)*mod_n - h
    # delta_w = ((w-1)//mod_n + 1)*mod_n - w
    # input_padded = torch.nn.functional.pad(input, (delta_w//2, delta_w-delta_w//2, delta_h//2, delta_h-delta_h//2))
    return input*2.-1.

# borrowed from https://github.com/daniilidis-group/neural_renderer/blob/master/neural_renderer/vertices_to_faces.py
def face_vertices_gen(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])

    nd = vertices.shape[2]
    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, nd))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]

def dict2obj(d):
    # if isinstance(d, list):
    #     d = [dict2obj(x) for x in d]
    if not isinstance(d, dict):
        return d

    class C(object):
        pass

    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o

class Pytorch3dRasterizer(nn.Module):
    """  Borrowed from https://github.com/facebookresearch/pytorch3d
    Notice:
        x,y,z are in image space, normalized
        can only render squared image now
    """

    def __init__(self, image_size=224):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        self.raster_settings_dict = {
            'image_size': image_size,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'bin_size': None,
            'max_faces_per_bin': None,
            'perspective_correct': False,
        }
        self.raster_settings = dict2obj(self.raster_settings_dict)

    def forward(self, vertices, faces, attributes=None):
        fixed_vertices = vertices.clone()
        fixed_vertices[..., :2] = -fixed_vertices[..., :2]
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        raster_settings = self.raster_settings
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
        )

        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone()
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        pixel_vals = torch.cat([pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)

        return pixel_vals, pix_to_face, bary_coords #, vismask

    def extra_repr(self):
        return '{image_size}px, blur_radius={blur_radius}, faces_per_pixel={faces_per_pixel}'.format(
            **self.raster_settings_dict)


def get_embedder(multires, i=1, use_gauss_encoding=False, gauss_sigma=1.0):
    if use_gauss_encoding:
        embed = rff.layers.GaussianEncoding(
            sigma=gauss_sigma, input_size=i, encoded_size=multires
        )
        print("Using random fourier gaussain encoding")
        return embed, 2 * multires
    else:
        if i == -1:
            return nn.Identity(), 3

        embed_kwargs = {
            "include_input": True,
            "input_dims": i,
            "max_freq_log2": multires - 1,
            "num_freqs": multires,
            "log_sampling": True,
            "periodic_fns": [torch.sin, torch.cos],
        }

        embedder_obj = Embedder(**embed_kwargs)
        embed = lambda x, eo=embedder_obj: eo.embed(x)
        return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


# class Embedder(nn.Module):
#     def __init__(self, N_freqs, input_dims=3, include_input=True) -> None:
#         super().__init__()
#         self.log_sampling = True
#         self.periodic_fns = [torch.sin, torch.cos]
#         self.max_freq = N_freqs - 1
#         self.N_freqs = N_freqs
#         self.include_input = include_input
#         self.input_dims = input_dims
#         embed_fns = []
#         if self.include_input:
#             embed_fns.append(lambda x: x)

#         if self.log_sampling:
#             freq_bands = 2.**torch.linspace(0.,
#                                             self.max_freq, steps=self.N_freqs)
#         else:
#             freq_bands = torch.linspace(
#                 2.**0., 2.**self.max_freq, steps=self.N_freqs)

#         for freq in freq_bands:
#             for p_fn in self.periodic_fns:
#                 embed_fns.append(lambda x, p_fn=p_fn,
#                                  freq=freq: p_fn(x * freq))
#         self.embed_fns = embed_fns
#         self.dim_embeded = self.input_dims*len(self.embed_fns)

#     def forward(self, inputs, alpha = 10.):
#         output = torch.cat([fn(inputs) for fn in self.embed_fns], 2)
#         start = 0
#         # print(alpha)
#         # if self.include_input:
#         #     start = 1
#         # for i in range(output.shape[1]//2):
#         #     output[:, (2*i+start)*self.input_dims:(2*(i+1)+start)*self.input_dims] *= (1-math.cos(math.pi*(max(min(alpha-i, 1.), 0.))))*.5
#         return output