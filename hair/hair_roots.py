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
import scipy.spatial as b
from shapely.geometry import Polygon, Point
from typing import Union, Optional
from scipy.spatial import Delaunay
import math


def save_pc_as_obj(pts, fpath="debug_points.obj", rgb=None):
    """
    Dump a point cloud to an .obj file - readable by Meshlab / Blender.
    
    pts  : (N,3) torch.Tensor | np.ndarray  – xyz in world units
    rgb  : (N,3) torch.Tensor | np.ndarray  – 0-1 or 0-255 colours  (optional)

    Usage (inside pdb):
        >>> save_pc_as_obj(my_tensor, "/tmp/pc.obj")
    """

    # --- bring everything to CPU-numpy ---------------------------------
    if isinstance(pts, torch.Tensor):
        pts = pts.detach().cpu().float().numpy()
    if rgb is not None and isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().float().numpy()

    with open(fpath, "w") as f:
        if rgb is None:
            for x, y, z in pts:
                f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        else:
            # .obj stores colours after xyz ⇒ many viewers pick them up
            if rgb.max() > 1.0:         # allow either 0-1 or 0-255
                rgb = rgb / 255.0
            for (x, y, z), (r, g, b) in zip(pts, rgb):
                f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")

    print(f"✔ wrote {len(pts)} vertices -> {fpath}")


def _sample_uv_delaunay(uv_bounds, num_samples, max_iters=10000, tol=1e-6):
    """
    Helper: sample num_samples UVs *inside* [u0,u1]×[v0,v1]
    by incremental largest‐circumcircle insertion, always maintaining
    a valid Delaunay on the unit-square domain.
    Returns an (num_samples,2) array in the *original* UV coords.
    """
    u0, u1, v0, v1 = uv_bounds
    # forward/back mappings between original UV and [0,1]^2
    def to_unit(uv):
        return np.stack([(uv[:,0]-u0)/(u1-u0),
                         (uv[:,1]-v0)/(v1-v0)], axis=1)
    def from_unit(xy):
        return np.stack([u0 + xy[:,0]*(u1-u0),
                         v0 + xy[:,1]*(v1-v0)], axis=1)

    eps = 1e-3
    # seed the *corners* of your scalp_bounds (in original UV)
    corners_raw = np.array([
        [u0 + eps*(u1-u0), v0 + eps*(v1-v0)],
        [u1 - eps*(u1-u0), v0 + eps*(v1-v0)],
        [u1 - eps*(u1-u0), v1 - eps*(v1-v0)],
        [u0 + eps*(u1-u0), v1 - eps*(v1-v0)],
    ], dtype=np.float64)
    uv = list(to_unit(corners_raw))  # now in [0,1]^2

    def circumcentres_and_radii(pts, tris):
        A = pts[tris[:,0]]; B = pts[tris[:,1]]; C = pts[tris[:,2]]
        a = B - A; b = C - A
        a2 = (a*a).sum(axis=1); b2 = (b*b).sum(axis=1)
        cross = a[:,0]*b[:,1] - a[:,1]*b[:,0]
        mask = np.abs(cross) > 1e-12
        centres = np.zeros_like(A); radii = np.zeros(len(tris))
        if mask.any():
            fac = 0.5 / cross[mask]
            cx = ( b[mask,1]*a2[mask] - a[mask,1]*b2[mask] ) * fac + A[mask,0]
            cy = ( a[mask,0]*b2[mask] - b[mask,0]*a2[mask] ) * fac + A[mask,1]
            centres[mask] = np.stack([cx, cy],1)
            radii[mask]   = np.linalg.norm(centres[mask]-A[mask], axis=1)
        return centres, radii

    it = 0
    while len(uv) < num_samples and it < max_iters:
        it += 1
        pts = np.vstack(uv)
        tri = Delaunay(pts)
        centres, radii = circumcentres_and_radii(pts, tri.simplices)
        idx = np.argmax(radii)
        c   = centres[idx].clip(eps, 1-eps)
        # reject too-close in unit space
        if np.min(np.linalg.norm(pts - c[None], axis=1)) < tol:
            continue
        uv.append(c)

    if len(uv) < num_samples:
        print(f"[WARN] only got {len(uv)}/{num_samples} UVs after {it} iters")

    # map those back into your true UV rectangle
    return from_unit(np.vstack(uv)[:num_samples])

class HairRoots(nn.Module):
    def __init__(
        self,
        head_mesh: str,
        scalp_vertex_idxs: Optional[List[int]] = None,   # <-- list of vertex indices
        scalp_bounds: Optional[List[float]] = None,
        mesh_scale: float = 1.0,
        mesh_translate: np.array = None,
        mean_center_before_rigid: bool = True
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

        # Translate, then scale here. Meant to align with FLAME canonical space.
        if mean_center_before_rigid:
            self.head.vertices -= self.head.vertices.mean(axis=0)
        if mesh_translate is not None:
            self.head += mesh_translate
        self.head.vertices *= mesh_scale

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
    
    def _project_to_surface(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Project arbitrary 3-D points to the closest point on the head mesh.

        Args
        ----
        pts : (N,3) torch tensor (any device)

        Returns
        -------
        (N,3) tensor on the same device - each row lies on the mesh surface
        """
        pts_np = pts.detach().cpu().numpy()
        loc_np, _, _ = self.head.nearest.on_surface(pts_np)  # (N,3) float64
        loc = torch.from_numpy(loc_np).to(pts.device, dtype=torch.float32)
        return loc

    def _uniform_point_in_poly(self, poly: Polygon, n=1):
        """
        Rejection-sample `n` uniform random points inside a 2-D polygon.
        Very small helper – CPU only, OK for a handful of points.
        """
        minx, miny, maxx, maxy = poly.bounds
        pts = []
        while len(pts) < n:
            cand = Point(np.random.uniform(minx, maxx),
                        np.random.uniform(miny, maxy))
            if poly.contains(cand):
                pts.append([cand.x, cand.y])
        return np.asarray(pts, np.float32)                 # (n,2)
    
    def _make_uv_seed_grid(self, bounds, n0: int = 128, jitter: float = 0.25) -> np.ndarray:
        """
        Create a quasi-uniform blanket of UV seed points that already covers
        the scalp AABB.  Returned array is (K,2) in *absolute* UV, **float32**.

        Parameters
        ----------
        bounds : list/tuple [u_min, u_max, v_min, v_max]
        n0     : rough target number of seeds (64–256 is typical)
        jitter : random jitter as a fraction of the cell size
        """
        u0, u1, v0, v1 = bounds
        area  = (u1 - u0) * (v1 - v0)
        h     = np.sqrt(area / n0)          # square cell edge ≈ sqrt(area / n)
        n_u   = int(np.ceil((u1 - u0) / h))
        n_v   = int(np.ceil((v1 - v0) / h))

        us = u0 + (np.arange(n_u) + 0.5) * h
        vs = v0 + (np.arange(n_v) + 0.5) * h
        uu, vv = np.meshgrid(us, vs, indexing="xy")
        seeds  = np.stack([uu.ravel(), vv.ravel()], 1)

        # jitter to avoid a perfect grid (helps Delaunay robustness)
        jitter_amt = jitter * h
        seeds += np.random.uniform(-jitter_amt, jitter_amt, seeds.shape)

        # clip back into the AABB
        seeds[:, 0] = np.clip(seeds[:, 0], u0, u1)
        seeds[:, 1] = np.clip(seeds[:, 1], v0, v1)
        return seeds.astype(np.float32)

    @staticmethod
    def _circumcenters_and_radii(
        uv_pts: np.ndarray,           # (N,2) in [0,1]
        simplices: np.ndarray         # (T,3) triangle‐vertex indices
    ):
        """
        For each triangle in `simplices`, compute its circum‐centre and radius.
        Returns
        ----------
        centres : (T,2)  float32
        radii    : (T,)   float32
        """
        A = uv_pts[simplices[:,0]]
        B = uv_pts[simplices[:,1]]
        C = uv_pts[simplices[:,2]]
        a = B - A
        b = C - A
        a2 = np.sum(a*a, axis=1)
        b2 = np.sum(b*b, axis=1)
        cross = a[:,0]*b[:,1] - a[:,1]*b[:,0]
        # avoid degenerate
        mask = np.abs(cross) > 1e-12
        centres = np.zeros((simplices.shape[0],2), dtype=np.float32)
        radii   = np.zeros( simplices.shape[0],   dtype=np.float32)
        if mask.any():
            fac = 0.5 / cross[mask]
            cx = (  b[mask,1]*a2[mask] - a[mask,1]*b2[mask] ) * fac + A[mask,0]
            cy = (  a[mask,0]*b2[mask] - b[mask,0]*a2[mask] ) * fac + A[mask,1]
            centres[mask,0] = cx
            centres[mask,1] = cy
            diffs = centres[mask] - A[mask]
            radii[mask] = np.sqrt(np.sum(diffs*diffs, axis=1))
        return centres, radii

    def sample_scalp_mesh_delaunay(
        self,
        num_samples: int,
        pseudo_roots: Optional[torch.Tensor] = None,  # ignored
        max_iters: int = 10000,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        """
        Sample exactly `num_samples` roots by Delaunay‐driven UV insertion,
        strictly inside self.scalp_bounds, no repeats.
        """
        assert self.scalp_bounds is not None, "scalp_bounds must be set"
        # 1) get the UV positions in your original [u0,u1]×[v0,v1]
        uv_pts = _sample_uv_delaunay(self.scalp_bounds,
                                     num_samples,
                                     max_iters=max_iters,
                                     tol=tol)            # (N,2) numpy

        # 2) back to torch, normalize into [0,1]^2 for your spherical mapping
        uv = torch.from_numpy(uv_pts.astype(np.float32)) \
                  .to(self.centroid.device)

        # 3) ray‐cast each UV into 3D on the scalp
        xyz = self.spherical_to_cartesian(uv)         # (N,3)
        return xyz
    
    def densify_scalp_mesh_delaunay(
        self,
        roots: torch.Tensor,
        scale: float = 2.0,
        rounds: int = 2,
    ) -> torch.Tensor:
        """
        Fast but slightly less regular densification:
        add triangle centroids from successive Delaunay triangulations.

        Parameters
        ----------
        roots  : (N,3) torch tensor – original roots in XYZ.
        scale  : multiplicative density (>1). 2.0 doubles the number of roots.
                If scale<=1 the input is returned unchanged.
        rounds : how many centroid-insertion rounds (≥1).
                1 round is usually enough for up to ~2× density;
                2 rounds covers ~3×.

        Returns
        -------
        (⌈N·scale⌉,3) tensor, originals first, on the same device.
        """
        if scale <= 1.0:
            return roots

        dev      = roots.device
        target_n = int(np.ceil(len(roots) * scale))

        # --- UV of the existing roots (absolute, still in [0,1] range) ----
        with torch.no_grad():
            uv = self.cartesian_to_spherical(roots)[..., :2].cpu().numpy()

        u0, u1, v0, v1 = self.scalp_bounds
        def to_unit(uv_):
            return np.stack([(uv_[:, 0]-u0)/(u1-u0),
                            (uv_[:, 1]-v0)/(v1-v0)], 1)

        def from_unit(xy):
            return np.stack([u0 + xy[:, 0]*(u1-u0),
                            v0 + xy[:, 1]*(v1-v0)], 1)

        uv_unit = to_unit(uv)                 # seed list, shape (N,2)

        for _ in range(rounds):
            if len(uv_unit) >= target_n:
                break
            tri  = Delaunay(uv_unit)
            # triangle areas in unit square
            A = uv_unit[tri.simplices[:, 0]]
            B = uv_unit[tri.simplices[:, 1]]
            C = uv_unit[tri.simplices[:, 2]]
            areas = 0.5 * np.abs(
                (B[:, 0]-A[:, 0])*(C[:, 1]-A[:, 1]) -
                (B[:, 1]-A[:, 1])*(C[:, 0]-A[:, 0])
            )

            order = np.argsort(-areas)  # largest first
            for idx in order:
                if len(uv_unit) >= target_n:
                    break
                centroid = (A[idx] + B[idx] + C[idx]) / 3.0
                uv_unit = np.vstack([uv_unit, centroid])

        # ---- convert the *new* UVs back to XYZ ---------------------------
        uv_new_abs = torch.from_numpy(from_unit(uv_unit[len(roots):])).float().to(dev)
        xyz_new    = self.spherical_to_cartesian(uv_new_abs)
        return torch.cat([roots, xyz_new], 0)
    
    def sample_scalp_mesh(
        self,
        num_samples: int,
        pseudo_roots: torch.Tensor  # (M,3) arbitrary points near the scalp
    ) -> torch.Tensor:
        """
        1. Snap each pseudo_root to the closest point on the head surface,
        recording which face it landed on.
        2. Build a submesh consisting only of those faces.
        3. Return num_samples points:
        • all snapped roots (deduplicated, clipped),
        • plus uniform samples on that submesh to fill up to num_samples.
        """
        device = pseudo_roots.device

        # 1) project each pseudo_root onto the mesh surface
        pts_np = pseudo_roots.cpu().numpy()
        snapped_pts, _, face_idx = self.head.nearest.on_surface(pts_np)
        # snapped_pts: (M,3) numpy, face_idx: (M,) numpy

        # convert back to torch
        roots_on_surf = torch.from_numpy(snapped_pts.astype(np.float32)).to(device)

        # 2) build a submesh using only the faces hit by those projections
        unique_faces = np.unique(face_idx)
        submesh = trimesh.Trimesh(
            vertices=self.head.vertices,
            faces=self.head.faces[unique_faces],
            process=False
        )

        # 3a) deduplicate and clip the snapped roots
        unique_roots = np.unique(snapped_pts, axis=0).astype(np.float32)
        np.random.shuffle(unique_roots)
        clipped = unique_roots[:num_samples]
        pts_list     = [clipped]

        # 3b) if we need more, sample uniformly on that submesh
        remainder = num_samples - clipped.shape[0]
        if remainder > 0:
            extra_pts, _ = trimesh.sample.sample_surface_even(submesh, remainder)
            pts_list.append(extra_pts.astype(np.float32))

        # 3c) assemble and return
        all_pts = np.vstack(pts_list)
        assert all_pts.shape[0] == num_samples, f"Expected {num_samples} points but got {all_pts.shape[0]}"
        return torch.from_numpy(all_pts).to(device)
    
    def sample_scalp_mesh_old(
        self,
        num_samples: int,
        pseudo_roots: torch.Tensor  # shape (M,3)
    ) -> torch.Tensor:
        """
        Sample points uniformly on the scalp mesh, but only on faces:
        1) Whose vertices are in the scalp mask, AND
        2) Whose all-vertex y > min(pseudo_roots[:,1])

        Args:
            num_samples: how many points to sample
            pseudo_roots: a (M,3) tensor of points; y-threshold = pseudo_roots[:,1].min()

        Returns:
            Tensor of shape (num_samples, 3) of sampled points.
        """
        # 1) build the scalp-only face set
        faces = self.head.faces
        mask_idxs = np.array(list(self.scalp_vertex_idxs), dtype=int)
        mask_keep = np.all(np.isin(faces, mask_idxs), axis=1)
        scalp_faces = faces[mask_keep]

        # 2) derive y_min from your pseudo-roots
        if isinstance(pseudo_roots, torch.Tensor):
            y_vals = pseudo_roots[:, 1].cpu().numpy()
        else:
            y_vals = np.asarray(pseudo_roots)[:, 1]
        y_min = float(y_vals.min())

        # 3) cull any scalp face that has a vertex at or below y_min
        verts = self.head.vertices  # (V,3) numpy
        face_y = verts[scalp_faces][:, :, 1]  # (F,3), take column 1 for Y
        y_keep = np.all(face_y > y_min, axis=1)
        final_faces = scalp_faces[y_keep]

        # 4) build a little submesh
        submesh = trimesh.Trimesh(
            vertices=verts,
            faces=final_faces,
            process=False
        )

        # 5) sample uniformly across its surface
        pts, _ = trimesh.sample.sample_surface_even(submesh, num_samples)

        return torch.from_numpy(pts).float()

    def uv_to_cartesian(self, uv_norm: torch.Tensor) -> torch.Tensor:
        """
        Map UV ∈ [0,1]² back to 3-D scalp positions by ray-casting.
        (No pre-baked `scalp_xyz_map` needed; **not differentiable**.)

        Args
        ----
        uv_norm : (..., 2) tensor on any device, each (u,v) ∈ [0,1].

        Returns
        -------
        (..., 3) tensor on the *same* device with points on the head surface.
        """
        # ---------- 0. reshape & move to CPU for trimesh -------------
        extra  = uv_norm.shape[:-1]
        uv_np  = uv_norm.reshape(-1, 2).detach().cpu().numpy()        # (N,2)
        N      = uv_np.shape[0]

        # ---------- 1. build ray directions from (u,v) --------------
        uv_pi  = uv_np * np.pi
        cot_u, cot_v = 1.0 / np.tan(uv_pi[:, 0]), 1.0 / np.tan(uv_pi[:, 1])
        h      = 2.0 / (cot_u**2 + cot_v**2 + 1.0)
        dirs   = np.stack([h * cot_u, h - 1.0, h * cot_v], axis=1)    # (N,3)
        dirs   = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)   # unit-length

        # ---------- 2. shoot rays from the centroid -----------------
        origin = self.centroid.detach().cpu().numpy()
        origins = np.repeat(origin[None, :], N, axis=0)               # (N,3)

        pts, hit_idx, _ = self.head.ray.intersects_location(
            ray_origins     = origins,
            ray_directions  = dirs,
            multiple_hits   = False
        )

        # ---------- 3. allocate output & scatter hits ----------------
        out = np.repeat(origin[None, :], N, axis=0)                   # default: centroid
        if len(hit_idx):
            out[hit_idx] = pts

        # ---------- 4. nearest-surface fallback for misses -----------
        miss = np.setdiff1d(np.arange(N), hit_idx)
        if miss.size:
            # shoot the ray “far” and snap to nearest surface
            bounds   = self.head.bounds
            diameter = np.linalg.norm(bounds[1] - bounds[0])
            far_pts  = origins[miss] + dirs[miss] * diameter
            close_pts, _, _ = self.head.nearest.on_surface(far_pts)
            out[miss] = close_pts

        # ---------- 5. reshape & return on original device ----------
        xyz = torch.from_numpy(out.astype(np.float32)
                            ).to(uv_norm.device).reshape(*extra, 3)
        return xyz

    @torch.no_grad()
    def sample_scalp_hex(
        self,
        num_samples: int,
        pseudo_roots: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """Hex-tiling sampling (original logic) + optional culling via pseudo_roots."""

        # ─── Original sampling logic (UNTOUCHED) ────────────────────────────────
        assert self.scalp_bounds is not None, "Set `scalp_bounds` before calling."
        device = self.centroid.device

        u0, u1, v0, v1 = self.scalp_bounds
        width_uv, height_uv = u1 - u0, v1 - v0
        area_uv = width_uv * height_uv

        a = math.sqrt((2.0 / math.sqrt(3.0)) * (area_uv / float(num_samples)))
        self.current_hc_spacing = a
        hex_radius = a / math.sqrt(3.0)

        dx = a
        dy = (math.sqrt(3.0) / 2.0) * a
        n_cols = int(math.ceil(width_uv  / dx)) + 1
        n_rows = int(math.ceil(height_uv / dy)) + 1

        row_idxs = np.arange(n_rows, dtype=np.int64)
        col_idxs = np.arange(n_cols, dtype=np.int64)

        v_coords_full  = v0 + row_idxs.astype(np.float32) * dy
        valid_rows     = v_coords_full <= v1
        v_coords       = v_coords_full[valid_rows]
        actual_rows    = row_idxs[valid_rows]

        offsets = ((actual_rows % 2) * (dx / 2.0)).astype(np.float32)
        uu  = (u0 + col_idxs.astype(np.float32) * dx)[None, :]
        off = offsets[:, None]
        u_grid = uu + off
        v_grid = np.repeat(v_coords[:, None], n_cols, axis=1)

        uv_all = np.stack([u_grid.ravel(), v_grid.ravel()], axis=1)
        inside = (
            (uv_all[:, 0] >= u0) & (uv_all[:, 0] <= u1) &
            (uv_all[:, 1] >= v0) & (uv_all[:, 1] <= v1)
        )
        uv_in = uv_all[inside]

        if uv_in.shape[0] == 0:
            empty = torch.zeros((0, 3), device=device, dtype=torch.float32)
            return empty, hex_radius, empty

        if uv_in.shape[0] > num_samples:
            uv_sel = uv_in[:num_samples]
        else:
            uv_sel = uv_in

        uv_tensor = torch.from_numpy(uv_sel.astype(np.float32)).to(device)
        all_xyz = self.spherical_to_cartesian(uv_tensor)  # (N,3)

        culled_xyz = all_xyz  # keep everything by default
        if self.scalp_vertex_idxs is not None and len(self.scalp_vertex_idxs):
            # Cache the list of faces made exclusively of scalp-vertices
            if not hasattr(self, "_scalp_face_indices"):
                faces_np   = self.head.faces                 # (F,3)
                in_mask    = np.isin(faces_np, list(self.scalp_vertex_idxs))
                valid_mask = np.all(in_mask, axis=1)
                self._scalp_face_indices = np.where(valid_mask)[0]

            if len(self._scalp_face_indices):
                # For each candidate, find which face it projects onto
                xyz_np = all_xyz.detach().cpu().numpy()
                _, _, root_faces = self.head.nearest.on_surface(xyz_np)
                keep = np.isin(root_faces, self._scalp_face_indices)
                culled_np  = xyz_np[keep]
                culled_xyz = torch.from_numpy(culled_np.astype(np.float32)).to(device)

        return all_xyz, hex_radius, culled_xyz


    @torch.no_grad()
    def densify_scalp_hex(
        self,
        pseudo_roots: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """Finer hex-tiling densification (original logic) + optional pseudo-root culling."""

        # ─── Original densification logic (UNTOUCHED) ──────────────────────────
        assert hasattr(self, "current_hc_spacing"), "Call sample_scalp_hex() first."
        device = self.centroid.device

        a_old = self.current_hc_spacing
        a_new = a_old / 2.0
        self.current_hc_spacing = a_new
        hex_radius_new = a_new / math.sqrt(3.0)

        u0, u1, v0, v1 = self.scalp_bounds
        dx = a_new
        dy = (math.sqrt(3.0) / 2.0) * a_new
        width_uv, height_uv = u1 - u0, v1 - v0

        n_cols = int(math.ceil(width_uv  / dx)) + 1
        n_rows = int(math.ceil(height_uv / dy)) + 1

        row_idxs = np.arange(n_rows, dtype=np.int64)
        col_idxs = np.arange(n_cols, dtype=np.int64)

        v_coords_full  = v0 + row_idxs.astype(np.float32) * dy
        valid_rows     = v_coords_full <= v1
        v_coords       = v_coords_full[valid_rows]
        actual_rows    = row_idxs[valid_rows]

        offsets = ((actual_rows % 2) * (dx / 2.0)).astype(np.float32)
        uu  = (u0 + col_idxs.astype(np.float32) * dx)[None, :]
        off = offsets[:, None]
        u_grid = uu + off
        v_grid = np.repeat(v_coords[:, None], n_cols, axis=1)

        uv_all = np.stack([u_grid.ravel(), v_grid.ravel()], axis=1)
        inside = (
            (uv_all[:, 0] >= u0) & (uv_all[:, 0] <= u1) &
            (uv_all[:, 1] >= v0) & (uv_all[:, 1] <= v1)
        )
        uv_in = uv_all[inside]

        if uv_in.shape[0] == 0:
            empty = torch.zeros((0, 3), device=device, dtype=torch.float32)
            return empty, hex_radius_new, empty

        uv_tensor = torch.from_numpy(uv_in.astype(np.float32)).to(device)
        all_xyz = self.spherical_to_cartesian(uv_tensor)  # (M,3)

        # ─── Culling via pseudo_roots (NEW) ─────────────────────────────────────
        culled_xyz = all_xyz  # keep everything by default

        if self.scalp_vertex_idxs is not None and len(self.scalp_vertex_idxs):
            # Cache the list of faces made exclusively of scalp-vertices
            if not hasattr(self, "_scalp_face_indices"):
                faces_np   = self.head.faces                 # (F,3)
                in_mask    = np.isin(faces_np, list(self.scalp_vertex_idxs))
                valid_mask = np.all(in_mask, axis=1)
                self._scalp_face_indices = np.where(valid_mask)[0]

            if len(self._scalp_face_indices):
                # For each candidate, find which face it projects onto
                xyz_np = all_xyz.detach().cpu().numpy()
                _, _, root_faces = self.head.nearest.on_surface(xyz_np)
                keep = np.isin(root_faces, self._scalp_face_indices)
                culled_np  = xyz_np[keep]
                culled_xyz = torch.from_numpy(culled_np.astype(np.float32)).to(device)

        return all_xyz, hex_radius_new, culled_xyz
    @torch.no_grad()
    def sample_scalp_hex_cull(
        self,
        num_samples: int,
        pseudo_roots: Optional[torch.Tensor] = None  # used to define region
    ) -> Tuple[torch.Tensor, float]:
        """
        Build a hex-tiling over the UV AABB, cull any cells outside the submesh
        defined by `pseudo_roots` (snapped to the head surface), and return
        their 3D positions plus the hexagon circumradius in UV.
        """
        assert self.scalp_bounds is not None, "Must set `scalp_bounds` before calling sample_scalp_hex."
        device = self.centroid.device

        # -- 1) snap pseudo_roots to mesh and build submesh ------------------
        assert isinstance(pseudo_roots, torch.Tensor) and pseudo_roots.numel() > 0, \
            "`pseudo_roots` must be a nonempty (N,3) Tensor"
        pts_np = pseudo_roots.detach().cpu().numpy()
        snapped_pts, _, face_idx = self.head.nearest.on_surface(pts_np)
        valid_faces = np.unique(face_idx)
        submesh = trimesh.Trimesh(
            vertices=self.head.vertices,
            faces=self.head.faces[valid_faces],
            process=False
        )

        # -- 2) compute original hex grid in absolute UV ---------------------
        u0, u1, v0, v1 = self.scalp_bounds
        width_uv = u1 - u0
        height_uv = v1 - v0
        area_uv = width_uv * height_uv

        # triangular‐lattice spacing a so ≈ num_samples cells
        a = math.sqrt((2.0 / math.sqrt(3.0)) * (area_uv / float(num_samples)))
        self.current_hc_spacing = a
        hex_radius = a / math.sqrt(3.0)

        dx = a
        dy = (math.sqrt(3.0) / 2.0) * a
        n_cols = int(math.ceil(width_uv  / dx)) + 1
        n_rows = int(math.ceil(height_uv / dy)) + 1

        row_idxs = np.arange(n_rows, dtype=np.int64)
        col_idxs = np.arange(n_cols, dtype=np.int64)

        v_coords_full = v0 + row_idxs.astype(np.float32) * dy
        valid_rows = v_coords_full <= v1
        v_coords = v_coords_full[valid_rows]
        actual_rows = row_idxs[valid_rows]

        offsets = ((actual_rows % 2) * (dx/2.0)).astype(np.float32)
        uu = (u0 + col_idxs.astype(np.float32)*dx)[None, :]
        off = offsets[:, None]
        u_grid = uu + off
        v_grid = np.repeat(v_coords[:, None], n_cols, axis=1)

        uv_all = np.stack([u_grid.ravel(), v_grid.ravel()], axis=1)
        inside = (
            (uv_all[:,0] >= u0) & (uv_all[:,0] <= u1) &
            (uv_all[:,1] >= v0) & (uv_all[:,1] <= v1)
        )
        uv_sel = uv_all[inside]

        # -- 3) project each UV to Cartesian & cull by submesh membership ---
        uv_tensor = torch.from_numpy(uv_sel.astype(np.float32)).to(device)
        xyz_pred = self.spherical_to_cartesian(uv_tensor)
        xyz_np = xyz_pred.detach().cpu().numpy()

        # find closest-face for each candidate and keep those on submesh
        _, _, cand_face = submesh.nearest.on_surface(xyz_np)
        keep = np.isin(cand_face, np.arange(len(valid_faces)))

        final_xyz = torch.from_numpy(xyz_np[keep].astype(np.float32)).to(device)
        # truncate so we return at most num_samples
        if len(final_xyz) > num_samples:
            final_xyz = final_xyz[:num_samples]

        return final_xyz, hex_radius

    @torch.no_grad()
    def densify_scalp_hex_cull(
        self,
        old_roots: Optional[torch.Tensor] = None  # ignored, because we re‐tile from scratch
    ) -> Tuple[torch.Tensor, float]:
        """
        As before, build a finer honeycomb grid over the same AABB,
        but *then* cull any hex‐vertices whose UV falls outside the
        triangulated UV‐hull of the current roots.
        """
        assert hasattr(self, "current_hc_spacing"), "You must call sample_scalp_honeycomb(...) first."
        u0, u1, v0, v1 = self.scalp_bounds
        device = self.centroid.device

        # 1) Halve spacing & recompute hex‐circumradius
        a_old = self.current_hc_spacing
        a_new = a_old / 2.0
        self.current_hc_spacing = a_new
        hex_radius_new = a_new / math.sqrt(3.0)

        # 2) Build absolute‐UV triangular grid (as before)
        dx = a_new
        dy = (math.sqrt(3.0) / 2.0) * a_new
        width_uv, height_uv = u1 - u0, v1 - v0
        n_cols = int(math.ceil(width_uv  / dx)) + 1
        n_rows = int(math.ceil(height_uv / dy)) + 1

        row_idxs = np.arange(n_rows, dtype=np.int64)
        col_idxs = np.arange(n_cols, dtype=np.int64)
        v_coords_full  = v0 + row_idxs.astype(np.float32) * dy
        valid_row_mask = (v_coords_full <= v1)
        v_coords       = v_coords_full[valid_row_mask]
        actual_rows    = row_idxs[valid_row_mask]

        offsets_for_row = ((actual_rows % 2) * (dx / 2.0)).astype(np.float32)

        uu     = (u0 + col_idxs.astype(np.float32) * dx)[None, :]
        off    = offsets_for_row[:, None]
        u_grid = uu + off
        v_grid = np.repeat(v_coords[:, None], n_cols, axis=1)

        u_flat = u_grid.reshape(-1)
        v_flat = v_grid.reshape(-1)
        uv_all = np.stack([u_flat, v_flat], axis=1)  # (R*n_cols, 2)

        # 3) First cull by AABB
        keep = (
            (uv_all[:,0] >= u0) & (uv_all[:,0] <= u1) &
            (uv_all[:,1] >= v0) & (uv_all[:,1] <= v1)
        )
        uv_in = uv_all[keep]  # still in absolute UV

        # 4) New: cull by current‐roots' UV‐triangulation
        if len(uv_in) > 0:
            # compute absolute-UV of your existing roots
            with torch.no_grad():
                uv_roots = self.cartesian_to_spherical(self.roots)[..., :2]  # (N_roots,2)
            uv_roots_np = uv_roots.cpu().numpy().astype(np.float32)

            tri = Delaunay(uv_roots_np)
            mask = tri.find_simplex(uv_in) >= 0
            uv_in = uv_in[mask]

        # 5) If nothing remains, return empty
        if uv_in.shape[0] == 0:
            return torch.zeros((0,3), device=device, dtype=torch.float32), hex_radius_new

        # 6) Project the survivors back to 3D
        uv_tensor = torch.from_numpy(uv_in.astype(np.float32)).to(device)
        xyz_new   = self.spherical_to_cartesian(uv_tensor)

        return xyz_new, hex_radius_new