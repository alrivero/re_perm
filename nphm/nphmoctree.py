import torch
import trimesh
import numpy as np
import plotly.graph_objects as go
import torch.nn.functional as F
from pathlib import Path

from typing import Optional, Union, Tuple
from tqdm import tqdm
from tqdm import trange
from kaolin.ops.spc.spc import (
    feature_grids_to_spc,
    scan_octrees,
    generate_points,
    to_dense
)
from skimage.measure import marching_cubes as mc

def get_sdf_and_normals(
    decoder,
    encoding: dict,
    grid_points: torch.Tensor,
    nbatch_points: int = 50_000
) -> torch.Tensor:
    """
    Query the decoder at grid_points (shape (1, N, 3)) and return
    an (N,4) numpy array where [:,0] is the SDF and [:,1:4] are the gradients.
    """
    device = encoding['geo'].device

    # Normalize to (1, N, 3)
    pts = grid_points
    if pts.ndim == 2 and pts.shape[1] == 3:
        pts = pts.unsqueeze(0)
    elif pts.ndim == 2 and pts.shape[0] == 3:
        pts = pts.T.unsqueeze(0)
    elif not (pts.ndim == 3 and pts.shape[0] == 1 and pts.shape[2] == 3):
        raise ValueError(f"Expected (N,3),(3,N) or (1,N,3), got {tuple(pts.shape)}")

    pts = pts.to(device).clone()
    chunks = torch.split(pts, nbatch_points, dim=1)
    out_list = []

    # Wrap the loop with tqdm
    for c in tqdm(chunks, desc="Sampling SDF & normals", total=len(chunks)):
        in_dict = {'queries': c}
        # if your id_model needs anchors, supply them:
        if hasattr(decoder, 'id_model') and hasattr(decoder.id_model, 'mlp_pos') and decoder.id_model.mlp_pos is not None:
            in_dict['anchors'] = decoder.id_model.get_anchors(encoding['geo'])
        out = decoder(in_dict, encoding, return_grad=True, ignore_deformations=True)
        sdf  = out['sdf'].squeeze(0).detach()      # (M,1)
        grad = out['gradient'].squeeze(0).detach() # (M,3)
        out_list.append(torch.cat([sdf, grad], dim=1).cpu())
        torch.cuda.empty_cache()

    all_feats = torch.cat(out_list, dim=0)  # (N,4)
    return all_feats.numpy()


class NPHMOctree:
    """
    Builds and holds an SPC‐octree of [sdf, nx, ny, nz].
    """

    def __init__(self,
        decoder,
        encoding: dict,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
        scale: float    = 4.0,
        resolution: int = 256,
        device: str     = 'cuda'
    ):
        self.resolution = resolution
        self.scale      = scale
        self.device     = device

        # 1) build a (R,R,R,3) world‐space grid
        aabb_min = aabb_min.to(device)
        aabb_max = aabb_max.to(device)
        xs = torch.linspace(aabb_min[0], aabb_max[0], resolution, device=device)
        ys = torch.linspace(aabb_min[1], aabb_max[1], resolution, device=device)
        zs = torch.linspace(aabb_min[2], aabb_max[2], resolution, device=device)
        grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing='ij'), dim=-1)  # (R,R,R,3)

        center = 0.5 * (aabb_min + aabb_max)          # shape (3,)
        center = center.view(1, 1, 1, 3)              # broadcast with grid
        pts_cano = ((grid - center) * scale + (scale * center)) \
                    .view(-1, 3) \
                    .unsqueeze(0)

        # 3) sample SDF + normals
        feats_np = get_sdf_and_normals(decoder, encoding, pts_cano)  # (N,4)
        feats = torch.from_numpy(feats_np).to(device) \
                          .view(resolution, resolution, resolution, 4)

        # 4) pack into dense feature grid (1,4,D,H,W)
        feature_grid = feats.permute(3,0,1,2).unsqueeze(0)

        # 5) convert to SPC octree
        self.octrees, self.lengths, self.features = feature_grids_to_spc(feature_grid)
        self.max_depth, self.pyramids, self.exsum = scan_octrees(self.octrees, self.lengths)

        # record AABB for later
        self.aabb_min = aabb_min
        self.aabb_max = aabb_max


    def sample(self, query_pts: torch.Tensor, depth=None, phys_scale=1.00):
        if depth is None:
            depth = self.max_depth
        B, N, _ = query_pts.shape

        # --------------------- world → voxel index → [-1,1] --------------------
        pts_unscaled = query_pts / phys_scale
        D   = 2 ** depth
        tmp = (pts_unscaled - self.aabb_min) / (self.aabb_max - self.aabb_min) * (D - 1)
        grid_coords = (tmp / (D - 1) * 2 - 1).view(B, N, 1, 1, 3)   # (x,y,z) still (X,Y,Z)

        # --------------------- dense volume (permute!) ------------------------
        hier  = generate_points(self.octrees, self.pyramids, self.exsum)
        dense = to_dense(hier, self.pyramids, self.features, depth)          # (1,C,X,Y,Z)
        dense = dense.permute(0, 1, 4, 3, 2)   #  **NEW**  -> (1,C,Z,Y,X) to match grid_sample

        # --------------------- trilinear sample -------------------------------
        samp = F.grid_sample(dense.expand(B,-1,-1,-1,-1),
                            grid_coords, mode='bilinear', align_corners=True)
        samp = samp.squeeze(-1).squeeze(-1)      # (B,4,N)

        sdfs    = samp[:, 0] * phys_scale        # (B,N)
        normals = samp[:, 1:4].permute(0, 2, 1)  # (B,N,3)   already in world axes
        return sdfs, normals
    
    def _sample_sdf_grid(self):
        """
        Internal: reconstruct the full octree‐resolution SDF volume at self.max_depth.
        Returns:
          xs, ys, zs: 1D numpy arrays of length D
          sdf:        (D, D, D) numpy array of SDF values
        """
        # 1) Reconstruct the full dense grid at the chosen depth:
        hier  = generate_points(self.octrees, self.pyramids, self.exsum)
        dense = dense = to_dense(hier,
                        self.pyramids,
                        self.features,
                        self.max_depth
        )
        # dense.shape == (1, 4, D, D, D)
        D = dense.shape[-1]

        # 2) Build world‐space coordinates arrays of length D:
        aabb_min, aabb_max = self.aabb_min.cpu().numpy(), self.aabb_max.cpu().numpy()
        xs = np.linspace(aabb_min[0], aabb_max[0], D)
        ys = np.linspace(aabb_min[1], aabb_max[1], D)
        zs = np.linspace(aabb_min[2], aabb_max[2], D)

        # 3) Create a normalized grid for trilinear interpolation in [-1,1]^3:
        grid_coords = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, D, device=self.device),
            torch.linspace(-1, 1, D, device=self.device),
            torch.linspace(-1, 1, D, device=self.device),
            indexing='ij'
        ), dim=-1).unsqueeze(0)  # (1, D, D, D, 3)

        # 4) Sample the dense volume:
        sampled = F.grid_sample(
            dense,              # (1, 4, D, D, D)
            grid_coords,        # (1, D, D, D, 3)
            mode='bilinear',
            align_corners=True
        )

        # 5) Extract the SDF channel and reshape to (D, D, D):
        vol = sampled.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()  # (D, D, D, 4)
        sdf = vol[..., 0]  # just the first channel

        return xs, ys, zs, sdf
    
    def export_slice_slider(
        self,
        html_path: str = "sdf_slices.html",
        axis: str      = "x",          # ←  'x' | 'y' | 'z'
        cmap: str      = "RdBu",
        vlim: Tuple[float, float] = None
    ):
        xs, ys, zs, sdf = self._sample_sdf_grid()
        vmin, vmax = (sdf.min(), sdf.max()) if vlim is None else vlim

        # pick the slicing direction
        if axis == "z":
            coord, u, v = zs, xs, ys
            slice2d     = lambda k: sdf[:, :, k].T          # (y,x) so T for img coords
            title_fmt   = "z = {:.3f}"
        elif axis == "y":
            coord, u, v = ys, xs, zs
            slice2d     = lambda k: sdf[:, k, :].T
            title_fmt   = "y = {:.3f}"
        elif axis == "x":
            coord, u, v = xs, ys, zs
            slice2d     = lambda k: sdf[k, :, :].T
            title_fmt   = "x = {:.3f}"
        else:
            raise ValueError("axis must be 'x', 'y' or 'z'")

        # first frame
        heat0 = go.Heatmap(z=slice2d(0), x=u, y=v,
                        zmin=vmin, zmax=vmax, colorscale=cmap,
                        colorbar=dict(title="SDF"))

        # animation frames
        frames = [
            go.Frame(
                data=[go.Heatmap(z=slice2d(k), x=u, y=v,
                                zmin=vmin, zmax=vmax, colorscale=cmap)],
                name=str(k),
                layout=go.Layout(title_text=title_fmt.format(coord[k])))
            for k in range(len(coord))
        ]

        # slider steps
        steps = [
            dict(method="animate", label=f"{coord[k]:.3f}",
                args=[[str(k)],
                    dict(mode="immediate",
                            frame=dict(duration=0, redraw=True),
                            transition=dict(duration=0))])
            for k in range(len(coord))
        ]

        fig = go.Figure(
            data=[heat0],
            frames=frames,
            layout=go.Layout(
                title=title_fmt.format(coord[0]),
                xaxis=dict(title=axis.replace('x', 'y').replace('y', 'z').replace('z', 'x')),  # any label you like
                yaxis=dict(title=""),
                updatemenus=[dict(type="buttons", showactive=False,
                                buttons=[dict(label="Play", method="animate",
                                                args=[None,
                                                    dict(frame=dict(duration=100, redraw=True),
                                                        fromcurrent=True)])])],
                sliders=[dict(active=0, pad={"t": 50}, steps=steps)]
            )
        )
        fig.write_html(html_path)
        print(f"→ slice slider exported to {html_path}")

    def visualize_band(self,
                    eps_band: float = 0.004,        # 4 mm half-width
                    depth: int = None,              # None → deepest level
                    html_path: str = "sdf_band.html"):
        """
        Interactively display the 0-iso surface and ±eps_band offsets.
        """
        # ------------------------------------------------ densify ----------
        if depth is None:
            depth = self.max_depth
        hier  = generate_points(self.octrees, self.pyramids, self.exsum)
        dense = to_dense(hier, self.pyramids, self.features, depth)   # (1,4,D,D,D)
        sdf   = dense[0, 0].cpu().numpy()                             # (D,D,D)
        D     = sdf.shape[0]

        # world-space voxel size & shift
        voxel = (self.aabb_max - self.aabb_min).cpu().numpy() / (D - 1)
        shift = self.aabb_min.cpu().numpy()

        def iso_mesh(level, color, opacity):
            v, f, _, _ = marching_cubes(sdf, level=level, spacing=(1, 1, 1))
            v = v * voxel + shift                                    # voxel → world
            return go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                            i=f[:, 0], j=f[:, 1], k=f[:, 2],
                            color=color, opacity=opacity,
                            flatshading=True, name=f"iso {level:+.3f}")

        mesh_zero  = iso_mesh(0.0,        "lightgrey", 0.4)
        mesh_outer = iso_mesh(+eps_band,  "green",     0.25)
        mesh_inner = iso_mesh(-eps_band,  "red",       0.25)

        fig = go.Figure(data=[mesh_zero, mesh_outer, mesh_inner])
        fig.update_layout(scene=dict(aspectmode="data"),
                        title=f"SDF 0-surface and ±{eps_band*1000:.1f} mm band")
        fig.write_html(html_path)
        print(f"→ band visualisation saved to {html_path}")

    def export_volume_rendering(self,
                                html_path: str = "sdf_volume.html",
                                isomin: float = None,
                                isomax: float = None,
                                opacity_scale: float = 0.1,
                                surface_count: int = 20):
        xs, ys, zs, sdf = self._sample_sdf_grid()
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='xy')

        fig = go.Figure(data=go.Volume(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=sdf.flatten(),
            isomin=sdf.min() if isomin is None else isomin,
            isomax=sdf.max() if isomax is None else isomax,
            opacity=opacity_scale, surface_count=surface_count,
            colorscale='RdBu'
        ))
        fig.update_layout(scene=dict(aspectmode='data'))
        fig.write_html(html_path)
        print(f"→ volume rendering exported to {html_path}")

    def export_slice_with_grad(self,
                            html_path="sdf_grad_xy.html",
                            axis="x",           # 'x' | 'y' | 'z'
                            every=4,            # down-sampling factor for arrows
                            cmap="RdBu"):

        xs, ys, zs, vol = self._sample_sdf_grid()        # vol shape (D,D,D,4)
        sdf, gx, gy, gz = vol[...,0], vol[...,1], vol[...,2], vol[...,3]

        # choose slicing direction ------------------------------------------------
        if axis == "z":
            coord, u, v, gx2d, gy2d = zs, xs, ys, gx, gy
            slice2d = lambda k: sdf[:,:,k].T                # image needs (v,u)
            grad2d  = lambda k: (gx2d[:,:,k].T, gy2d[:,:,k].T)
            title   = lambda c: f"z = {c:.3f}"
        elif axis == "y":
            coord, u, v, gx2d, gz2d = ys, xs, zs, gx, gz
            slice2d = lambda k: sdf[:,k,:].T
            grad2d  = lambda k: (gx2d[:,k,:].T, gz2d[:,k,:].T)
            title   = lambda c: f"y = {c:.3f}"
        else:  # axis == "x"
            coord, u, v, gy2d, gz2d = xs, ys, zs, gy, gz
            slice2d = lambda k: sdf[k,:,:].T
            grad2d  = lambda k: (gy2d[k,:,:].T, gz2d[k,:,:].T)
            title   = lambda c: f"x = {c:.3f}"

        # helper to build one frame ----------------------------------------------
        def frame(k):
            zi   = slice2d(k)
            gxk, gyk = grad2d(k)

            # down-sample to reduce clutter
            ui, vi, gxi, gyi = zi[::every,::every], u[::every], v[::every], gxk[::every,::every], gyk[::every,::every]
            U, V = np.meshgrid(vi, ui, indexing='xy')
            g   = np.stack([gxi, gyi], -1)
            nrm = np.linalg.norm(g, axis=-1, keepdims=True) + 1e-8
            g   = g / nrm                                         # unit vectors

            # image + quiver layer
            return [
                go.Heatmap(z=zi, x=u, y=v, colorscale=cmap,
                        zmin=sdf.min(), zmax=sdf.max()),
                go.Cone(x=U.flatten(), y=V.flatten(), z=np.zeros_like(U).flatten(),
                        u=g[...,0].flatten(), v=g[...,1].flatten(), w=np.zeros_like(U).flatten(),
                        showscale=False, sizemode="scaled", sizeref=0.5)
            ]

        # first frame
        fig = go.Figure(data=frame(0), 
                        frames=[go.Frame(data=frame(k), name=str(k), 
                                        layout=go.Layout(title_text=title(coord[k])))
                                for k in range(len(coord))])

        # slider UI
        fig.update_layout(
            title=title(coord[0]),
            xaxis_title="x" if axis != "x" else "y",
            yaxis_title="y" if axis != "y" else "z",
            updatemenus=[dict(type="buttons", showactive=False,
                            buttons=[dict(label="Play", method="animate",
                                            args=[None, dict(frame=dict(duration=100, redraw=True),
                                                            fromcurrent=True)])])],
            sliders=[dict(active=0, pad={"t":50},
                        steps=[dict(method="animate", label=f"{coord[k]:.3f}",
                                    args=[[str(k)],
                                            dict(frame=dict(duration=0, redraw=True),
                                                mode="immediate",
                                                transition=dict(duration=0))])
                                for k in range(len(coord))])]
        )
        fig.write_html(html_path)
        print("→ gradient slice slider saved to", html_path)
        
    # ----------------------------------------------------------------------
    #   put this inside class NPHMOctree
    # ----------------------------------------------------------------------
    def visualize_outward_tolerance(self,
                                    outside_tol: float = 0.003,   # 3 mm
                                    depth: int = None,            # None → deepest
                                    html_path: str = "sdf_out_tol.html"):
        """
        Show two marching‐cubes meshes in one Plotly scene
        • grey  :  f(x)=0   (stored shell)
        • green :  f(x)=+outside_tol  (outer allowance)

        Parameters
        ----------
        outside_tol : outward slack (metres)
        depth       : octree level to densify (smaller number ⇒ faster preview)
        html_path   : where to save the interactive HTML
        """
        if depth is None:
            depth = self.max_depth

        # 1) densify the SDF at chosen depth ---------------------------------
        hier  = generate_points(self.octrees, self.pyramids, self.exsum)
        dense = to_dense(hier, self.pyramids, self.features, depth)  # (1,4,D,D,D)
        sdf   = dense[0, 0].cpu().numpy()                            # (D,D,D)
        D     = sdf.shape[0]

        # voxel→world scaling
        voxel = (self.aabb_max - self.aabb_min).cpu().numpy() / (D - 1)
        shift = self.aabb_min.cpu().numpy()

        def iso_mesh(level, color, opacity, name):
            v, f, _, _ = mc(sdf, level=level, spacing=(1, 1, 1))
            v_world    = v * voxel + shift
            return go.Mesh3d(x=v_world[:, 0], y=v_world[:, 1], z=v_world[:, 2],
                            i=f[:, 0], j=f[:, 1], k=f[:, 2],
                            color=color, opacity=opacity, flatshading=True, name=name)

        mesh_zero  = iso_mesh(0.0,           "lightgrey", 0.40, "iso 0")
        mesh_outer = iso_mesh(outside_tol,   "green",     0.25, f"iso +{outside_tol:.3f}")

        fig = go.Figure(data=[mesh_zero, mesh_outer])
        fig.update_layout(scene=dict(aspectmode="data"),
                        title=f"SDF 0-surface and +{outside_tol*1000:.1f} mm tolerance")
        fig.write_html(html_path)
        print(f"→ outward-tolerance visualisation saved to {html_path}")

    def export_mesh(
            self,
            level        : Optional[int]       = None,
            iso_value    : float               = 0.0,
            offset       : float               = 0.0,   # ← new outward (+) / inward (–) tolerance
            out_path     : Union[str, Path]    = "sdf_mc.obj",
            return_arrays: bool                = False
        ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Extract an iso-surface of the SDF via marching cubes.

        Parameters
        ----------
        level : int, optional
            Octree level to densify.  None → deepest populated level.
        iso_value : float
            Base SDF value whose surface to extract (default 0).
        offset : float
            Additional offset to apply (e.g. +0.003 for a 3 mm outward shell).
        out_path : str or pathlib.Path
            Mesh file to write (.obj, .ply, .stl, …).
        return_arrays : bool
            If True, return (V, F) arrays in addition to writing the file.

        Returns
        -------
        (vertices, faces) if ``return_arrays`` is True, else None.
        """
        level = self.max_depth if level is None else level
        if level > self.max_depth:
            raise ValueError(f"Octree goes only to level {self.max_depth}, "
                            f"but level {level} was requested.")

        # 1) densify -----------------------------------------------------------------
        hier  = generate_points(self.octrees, self.pyramids, self.exsum)
        dense = to_dense(hier, self.pyramids, self.features, level)   # (1,4,D,D,D)
        sdf   = dense[0, 0].cpu().numpy()                             # (D,D,D)

        # 2) marching cubes at iso + offset -----------------------------------------
        iso_target = iso_value + offset
        verts, faces, _, _ = mc(sdf, level=iso_target, spacing=(1, 1, 1))  # voxel coords

        # 3) voxel → world -----------------------------------------------------------
        D = sdf.shape[0]
        voxel_size = (self.aabb_max - self.aabb_min) / float(D - 1)        # (3,)
        verts_world = (torch.as_tensor(verts.copy()).to(voxel_size.device)
                    * voxel_size + self.aabb_min).cpu().numpy()

        # 4) write mesh --------------------------------------------------------------
        out_path = Path(out_path)
        trimesh.Trimesh(verts_world, faces, process=False).export(out_path)

        if return_arrays:
            return verts_world, faces

    # ------------------------------------------------------------------
    def visualize_flow(self,
                    depth: int = None,
                    n_arrows: int = 4000,
                    default_scale: float = 0.2,
                    html_path: str = "sdf_flow.html"):
        """
        Visualise the SDF field: grey 0-iso surface plus cones showing
        outward normals (blue) and one tangent-plane direction (orange)
        sampled *throughout the volume*.

        depth        : octree level to densify (None → deepest)
        n_arrows     : number of random voxels to display
        default_scale: initial cone length (Plotly sizeref)
        """

        import plotly.graph_objects as go
        from skimage.measure import marching_cubes
        import numpy as np, torch

        depth = self.max_depth if depth is None else depth

        # ---------- densify at chosen depth ---------------------------------
        hier  = generate_points(self.octrees, self.pyramids, self.exsum)
        dense = to_dense(hier, self.pyramids, self.features, depth)   # (1,4,D,D,D)
        sdf   = dense[0, 0].cpu().numpy()                             # (D,D,D)
        grads = dense[0, 1:4].permute(1, 2, 3, 0).cpu().numpy()       # (D,D,D,3)

        D = sdf.shape[0]
        voxel = (self.aabb_max - self.aabb_min).cpu().numpy() / (D - 1)
        shift = self.aabb_min.cpu().numpy()

        # ---------- 0-iso mesh for context ----------------------------------
        v, f, _, _ = marching_cubes(sdf, level=0.0, spacing=(1,1,1))
        v_world = v * voxel + shift
        mesh = go.Mesh3d(x=v_world[:,0], y=v_world[:,1], z=v_world[:,2],
                        i=f[:,0], j=f[:,1], k=f[:,2],
                        color="lightgrey", opacity=0.35,
                        flatshading=True, name="0-iso", legendgroup="mesh")

        # ---------- random voxel samples ------------------------------------
        rng   = np.random.default_rng(0)
        total_vox = D**3
        pick_idx  = rng.choice(total_vox, size=min(n_arrows, total_vox), replace=False)

        # indices → (z,y,x) voxel coords
        k = pick_idx
        z = k % D
        y = (k // D) % D
        x = k // (D*D)
        pts_vox = np.stack([x, y, z], axis=1)         # (M,3)

        # voxel → world
        pts_w = pts_vox * voxel + shift               # (M,3)

        # normals from grid (already unit when stored)
        normals = grads[x, y, z]                      # (M,3)

        # ---------- tangent-plane vectors -----------------------------------
        up = np.array([0., 1., 0.])
        t_raw = np.cross(normals, up)
        bad = np.linalg.norm(t_raw, axis=1) < 1e-6
        if bad.any():
            alt = np.array([1., 0., 0.])
            t_raw[bad] = np.cross(normals[bad], alt)
        tangents = t_raw / (np.linalg.norm(t_raw, axis=1, keepdims=True) + 1e-9)

        # ---------- cone traces ---------------------------------------------
        cones_norm = go.Cone(x=pts_w[:,0], y=pts_w[:,1], z=pts_w[:,2],
                            u=normals[:,0], v=normals[:,1], w=normals[:,2],
                            colorscale=[[0,"blue"],[1,"blue"]],
                            anchor="tail", sizemode="scaled",
                            sizeref=default_scale, showscale=False,
                            name="normals", legendgroup="norm")

        cones_tan  = go.Cone(x=pts_w[:,0], y=pts_w[:,1], z=pts_w[:,2],
                            u=tangents[:,0], v=tangents[:,1], w=tangents[:,2],
                            colorscale=[[0,"orange"],[1,"orange"]],
                            anchor="tail", sizemode="scaled",
                            sizeref=default_scale, showscale=False,
                            name="tangent dir", legendgroup="tan")

        # ---------- slider to adjust arrow size -----------------------------
        scale_vals = [1.0, 2.0, 4.0, 8.0, 16.0]
        steps = [dict(method="restyle",
                    label=f"{s:.2f}",
                    args=[{"sizeref":[s, s]}, [1, 2]])
                for s in scale_vals]
        slider = [dict(
            active=scale_vals.index(default_scale)
                if default_scale in scale_vals else 2,
            currentvalue={"prefix":"arrow scale: "},
            steps=steps, pad={"t":40})]

        # ---------- assemble figure -----------------------------------------
        fig = go.Figure(data=[mesh, cones_norm, cones_tan])
        fig.update_layout(title=f"SDF flow (depth {depth}, {len(pick_idx)} arrows)",
                        scene=dict(aspectmode="data"),
                        sliders=slider)
        fig.write_html(html_path)
        print(f"→ flow visualisation saved to {html_path}")