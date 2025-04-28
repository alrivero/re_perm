from typing import List, Optional, Tuple

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import trimesh

from skimage.draw import polygon          
from typing import List, Optional, Tuple
from utils.misc import EPSILON
from utils.misc import copy2cpu as c2c
from torch import nn


class HairRoots(nn.Module):
    def __init__(
        self,
        head_mesh: str,
        scalp_vertex_idxs: Optional[List[int]] = None,   # <-- list of vertex indices
        scalp_bounds: Optional[List[float]] = None,
        mask_resolution: int = 256                    # <-- UV-mask size (H = W)
    ) -> None:
        """
        Args:
            head_mesh (str)              : Path to the head mesh (any format Trimesh loads).
            scalp_bounds (List[float]|None): Override for UV AABB [u_min,u_max,v_min,v_max].
            scalp_vertex_idxs (List[int]|None): Indices of vertices that belong to the scalp.
            mask_resolution (int)        : Resolution (height = width) for the scalp UV mask.
        """
        super().__init__()

        # ── load mesh & build centroid ───────────────────────────────
        self.head = trimesh.load(head_mesh)
        centroid = (self.head.bounds[0] + self.head.bounds[1]) / 2.0
        centroid[1] = 0.0            # keep head centred on y = 0
        self.register_buffer(       # <-- buffer ⇒ auto-moved
            "centroid",
            torch.as_tensor(centroid, dtype=torch.float32)
        )

        # ── vertex indices & UV bounds ───────────────────────────────
        if scalp_vertex_idxs is not None and len(scalp_vertex_idxs) > 0:
            # ensure Python ints so they’re hashable
            self.scalp_vertex_idxs = set(map(int, scalp_vertex_idxs))
        else:
            self.scalp_vertex_idxs = None

        self.scalp_bounds = scalp_bounds
        if self.scalp_vertex_idxs:
            self.scalp_bounds = self._compute_scalp_bounds()

        # ── 1-channel scalp mask (1,H,W)  ────────────────────────────
        if self.scalp_vertex_idxs:
            mask = self._generate_scalp_uv_mask(mask_resolution, mask_resolution)  # tensor
            self.register_buffer("scalp_mask", mask)
        else:
            self.register_buffer("scalp_mask", None)

        # ── differentiable UV ➜ XYZ lookup map  (1,3,H,W) ────────────
        if self.scalp_bounds is not None:
            uv_xyz   = self.uv(mask_resolution, mask_resolution, include_normal=False)   # (H,W,3) numpy
            xyz_map  = torch.from_numpy(uv_xyz.transpose(2, 0, 1)).unsqueeze(0)          # (1,3,H,W)
            self.register_buffer("scalp_xyz_map", xyz_map.float())
        else:
            self.register_buffer("scalp_xyz_map", None)

    def _compute_scalp_bounds(self) -> List[float]:
        """Return [u_min, u_max, v_min, v_max] for the supplied scalp vertices."""
        verts      = torch.as_tensor(self.head.vertices, dtype=torch.float32)          
        scalp_pos  = verts[list(self.scalp_vertex_idxs)]                                  
        uv_sphere  = self.cartesian_to_spherical(scalp_pos)[..., :2]                   
        u_min, v_min = uv_sphere.min(0).values
        u_max, v_max = uv_sphere.max(0).values
        return [float(u_min), float(u_max), float(v_min), float(v_max)]

    def _generate_scalp_uv_mask(
        self,
        height: int = 256,
        width:  int = 256
    ) -> torch.Tensor:
        """
        Rasterise every face whose three vertices are in `self.scalp_vertex_idxs`
        onto a boolean UV image of shape (height,width).  Returned tensor is
        (1,H,W) float32 with values 0/1.
        """
        assert self.scalp_bounds is not None, "Need scalp_bounds before making a mask"
        assert self.scalp_vertex_idxs,           "No scalp vertex indices supplied"

        # ---- pre-compute spherical-UV for *all* vertices (CPU, torch) -------------
        all_xyz   = torch.as_tensor(self.head.vertices, dtype=torch.float32)          
        all_uv    = self.cartesian_to_spherical(all_xyz)[..., :2]                     

        # ---- pick faces whose three verts ∈ scalp_vertex_idxs ------------------------
        faces = self.head.faces                                                     
        keep  = np.all(np.isin(faces, list(self.scalp_vertex_idxs)), axis=1)           
        faces = faces[keep]                                                         

        # ---- prepare blank mask --------------------------------------------------
        mask = np.zeros((height, width), dtype=np.float32)

        # ---- rescaling helpers ---------------------------------------------------
        u0, u1, v0, v1 = self.scalp_bounds
        def uv2pix(uv: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
            """map uv∈[bounds]→pixel coords"""
            u = (uv[..., 0] - u0) / (u1 - u0) * (width  - 1)
            v = (uv[..., 1] - v0) / (v1 - v0) * (height - 1)
            return u.numpy(), v.numpy()                                             

        # ---- rasterise every triangle -------------------------------------------
        for f in faces:
            tri_uv = all_uv[f]                                                      
            px, py = uv2pix(tri_uv)                                                 
            rr, cc = polygon(py, px, mask.shape)                                    
            mask[rr, cc] = 1.0

        # ---- CHW torch tensor in [0,1] -------------------------------------------
        return torch.from_numpy(mask).unsqueeze(0)                                  

    def cartesian_to_spherical(self, x: torch.Tensor) -> torch.Tensor:
        """ Parameterize the scalp surface by considering it as the upper half of a sphere.
        Reference: Wang, Lvdi, et al. "Example-based hair geometry synthesis." ACM SIGGRAPH 2009 papers

        Args:
            x (torch.Tensor): Cartesian points of shape (..., 3).

        Returns:
            (torch.Tensor): Spherical coordinates uvw of shape (..., 3).
        """
        if self.centroid.device != x.device:
            self.centroid = self.centroid.to(x.device)

        x_prime = x - self.centroid
        w = torch.norm(x_prime, dim=-1)
        p = x_prime / (w[..., None] + EPSILON)
        u = torch.acos(p[..., 0] / (p[..., 0] ** 2 + (p[..., 1] + 1) ** 2).sqrt()) / np.pi
        v = torch.acos(p[..., 2] / (p[..., 2] ** 2 + (p[..., 1] + 1) ** 2).sqrt()) / np.pi
        uvw = torch.stack([u, v, w], dim=-1)

        return uvw

    def spherical_to_cartesian(self, x: torch.Tensor) -> torch.Tensor:
        """ Remap spherical coordinates to Cartesian coordinates on the scalp, with nearest-point fallback.
        Reference: Wang, Lvdi, et al. "Example-based hair geometry synthesis." ACM SIGGRAPH 2009 papers

        Args:
            x (torch.Tensor): Spherical coordinates of shape (..., 2) or (..., 3).

        Returns:
            torch.Tensor: Cartesian coordinates xyz of shape (..., 3).
        """
        uv = x[..., :2] * np.pi
        cot_u = 1.0 / torch.tan(uv[..., 0])
        cot_v = 1.0 / torch.tan(uv[..., 1])

        h = 2 / (cot_u ** 2 + cot_v ** 2 + 1)
        p = torch.zeros(*uv.shape[:-1], 3, device=uv.device)
        p[..., 0] = h * cot_u
        p[..., 1] = h - 1
        p[..., 2] = h * cot_v

        # If radial component w is provided, use it directly
        if x.shape[-1] == 3:
            if self.centroid.device != x.device:
                self.centroid = self.centroid.to(x.device)
            return p * x[..., 2:].unsqueeze(-1) + self.centroid

        # Otherwise, ray-cast and fallback to nearest-point
        with torch.no_grad():
            # flatten rays
            extra_dims = p.shape[:-1]
            p_flat = p.reshape(-1, 3)
            R = p_flat.shape[0]

            # prepare origins at the centroid
            centroid = self.centroid.to(p_flat.device)
            origins = centroid.unsqueeze(0).expand(R, -1)

            # convert to CPU numpy for trimesh
            dirs_np = c2c(p_flat)
            origins_np = c2c(origins)

            # cast rays
            locations, index_ray, _ = self.head.ray.intersects_location(
                ray_origins=origins_np,
                ray_directions=dirs_np,
                multiple_hits=False
            )

            # prepare output buffer, defaulting to centroid
            xyz_flat = torch.zeros(R, 3, device=p.device, dtype=torch.float32)
            xyz_flat[:] = centroid

            # scatter hits
            if len(index_ray) > 0:
                hits = torch.tensor(locations, dtype=torch.float32, device=p.device)
                rays = torch.as_tensor(index_ray, dtype=torch.long, device=p.device)
                xyz_flat[rays] = hits

            # nearest-point fallback for misses
            all_rays = np.arange(R)
            hit_set = set(index_ray.tolist())
            miss_idx = np.array([i for i in all_rays if i not in hit_set], dtype=int)
            if miss_idx.size > 0:
                # estimate a far-away query point along each missed ray
                # use mesh diameter as distance
                bounds = self.head.bounds
                diameter = np.linalg.norm(bounds[1] - bounds[0])
                query_pts = origins_np[miss_idx] + dirs_np[miss_idx] * diameter
                closest_pts, _, _ = self.head.nearest.on_surface(query_pts)
                xyz_flat[miss_idx] = torch.tensor(closest_pts, dtype=torch.float32, device=p.device)

            # reshape back to original dims
            xyz = xyz_flat.reshape(*extra_dims, 3)
        return xyz

    def load_txt(self, fname: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Load root positions and normals from .txt files.

        Args:
            fname (str): File to load.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Root positions and normals.
        """
        data = np.loadtxt(fname, skiprows=1)
        position = torch.tensor(data[::2], dtype=torch.float32)
        normal = torch.tensor(data[1::2], dtype=torch.float32)

        return position, F.normalize(normal, dim=-1)

    def uv(self, width: int, height: int, include_normal: bool = True) -> np.ndarray:
        """Return an (H,W,3) or (H,W,6) UV-map, filling rays that miss the mesh
        with the nearest surface point (differentiable safety not required)."""
        assert self.scalp_bounds is not None, "AABB not set"

        # --- build the spherical directions exactly as before ---
        u, v = np.meshgrid(
            np.linspace(self.scalp_bounds[0], self.scalp_bounds[1], num=width),
            np.linspace(self.scalp_bounds[2], self.scalp_bounds[3], num=height),
            indexing='ij'
        )
        uv = np.stack([u, v], axis=-1).reshape(-1, 2)           # (R,2)
        uv_pi = uv * np.pi
        cot_u, cot_v = 1/np.tan(uv_pi[:,0]), 1/np.tan(uv_pi[:,1])
        h = 2 / (cot_u**2 + cot_v**2 + 1)
        dirs = np.stack([h*cot_u, h-1, h*cot_v], axis=1)        # (R,3)

        # --- cast all R rays at once ---
        origins = np.repeat(self.centroid.cpu().numpy()[None, :], dirs.shape[0], axis=0)
        loc, idx_ray, idx_tri = self.head.ray.intersects_location(
            ray_origins     = origins,
            ray_directions  = dirs,
            multiple_hits   = False
        )

        # --- allocate a full buffer & scatter the hits ---
        R = dirs.shape[0]
        pts = np.repeat(self.centroid.cpu().numpy()[None, :], R, axis=0)  # fallback
        if len(idx_ray):
            pts[idx_ray] = loc                                            # scatter hits

        # --- optional normals ---
        if include_normal and len(idx_ray):
            bary   = trimesh.triangles.points_to_barycentric(
                        self.head.triangles[idx_tri], loc)
            normals = self.head.vertex_normals[self.head.faces[idx_tri]]
            nrm     = trimesh.unitize((normals * bary[:, :, None]).sum(axis=1))
            full_n  = np.zeros_like(pts);  full_n[idx_ray] = nrm
            texture = np.concatenate([pts, full_n], axis=1)               # (R,6)
        else:
            texture = pts                                                 # (R,3)

        return texture.reshape(width, height, -1).transpose(1, 0, 2)       # (H,W,C)

    def surface_normals(self, points: np.ndarray, index_tri: np.ndarray) -> np.ndarray:
        """ Compute normals for points on the mesh surface.

        Args:
            points (np.ndarray): Points on the mesh surface of shape (n, 3).
            index_tri (np.ndarray): Triangle indices associated with points, of shape (n,).

        Returns:
            (np.ndarray): Surface normals of shape (n, 3).
        """
        bary = trimesh.triangles.points_to_barycentric(triangles=self.head.triangles[index_tri], points=points)
        normals = self.head.vertex_normals[self.head.faces[index_tri]]

        return trimesh.unitize((normals * bary.reshape((-1, 3, 1))).sum(axis=1))

    def bounds(self, roots: torch.Tensor) -> None:
        """ Compute AABB of all 2D hair roots in the dataset.

        Args:
            roots (torch.Tensor): Hair roots uv of shape (..., 2).
        """
        u_min = roots[..., 0].min()
        u_max = roots[..., 0].max()
        v_min = roots[..., 1].min()
        v_max = roots[..., 1].max()

        self.scalp_bounds = [u_min, u_max, v_min, v_max]
        print(self.scalp_bounds)

    def rescale(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        u0, u1, v0, v1 = self.scalp_bounds
        if inverse:
            # map [0,1] back to original UV bounds
            u = x[..., 0] * (u1 - u0) + u0
            v = x[..., 1] * (v1 - v0) + v0
        else:
            # normalize into [0,1]
            u = (x[..., 0] - u0) / (u1 - u0)
            v = (x[..., 1] - v0) / (v1 - v0)

        # clamp *out-of-place* (functional)
        u = u.clamp(min=0.0, max=1.0)
        v = v.clamp(min=0.0, max=1.0)

        if x.shape[-1] == 3:
            # preserve radial component
            w = x[..., 2]
            return torch.stack((u, v, w), dim=-1)
        else:
            return torch.stack((u, v), dim=-1)

    def sample_scalp_uv(self, num_samples: int) -> torch.Tensor:
        """
        Uniformly sample `num_samples` UV coords inside the scalp_bounds.

        Returns:
            uv_norm (torch.Tensor[num_samples,2])
              — normalized UVs in [0,1]^2
        """
        assert self.scalp_bounds is not None, "Must have scalp_bounds to sample UVs"
        u0, u1, v0, v1 = self.scalp_bounds
        device = self.centroid.device

        # sample raw UV in [u0,u1] x [v0,v1]
        u_raw = torch.rand(num_samples, device=device) * (u1 - u0) + u0
        v_raw = torch.rand(num_samples, device=device) * (v1 - v0) + v0
        uv_raw = torch.stack([u_raw, v_raw], dim=1)

        # normalize back to [0,1]^2
        uv_norm = torch.empty_like(uv_raw)
        uv_norm[:, 0] = (uv_raw[:, 0] - u0) / (u1 - u0)
        uv_norm[:, 1] = (uv_raw[:, 1] - v0) / (v1 - v0)
        return uv_norm


    def uv_to_cartesian(self, uv_norm: torch.Tensor) -> torch.Tensor:
        """
        Differentiable inversion of UV coords to 3D points on the scalp using bilinear sampling.

        Args:
            uv_norm (torch.Tensor[...,2]): UV coords in [0,1]^2
        Returns:
            torch.Tensor[...,3]: 3D points on the scalp surface
        """
        assert self.scalp_xyz_map is not None, "Need precomputed scalp_xyz_map"
        
        # flatten and prepare grid for sampling
        orig_shape = uv_norm.shape[:-1]
        uv_flat = uv_norm.reshape(-1,2).clamp(0.0,1.0)
        # convert to normalized grid coords [-1,1]
        grid = uv_flat * 2.0 - 1.0
        N = grid.shape[0]
        grid = grid.view(1,1,N,2)

        # sample: scalp_xyz_map (1,3,H,W)
        sampled = torch.nn.functional.grid_sample(
            self.scalp_xyz_map,
            grid,
            mode='bilinear',
            align_corners=True
        )  # (1,3,1,N)

        pts = sampled.view(3, N).transpose(0,1)  # (N,3)
        return pts.reshape(*orig_shape, 3)
