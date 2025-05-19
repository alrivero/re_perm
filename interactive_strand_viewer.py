#!/usr/bin/env python3
"""
Interactive Strand Viewer  ▸  GPU-accelerated, works on macOS.

Usage
-----
python interactive_strand_viewer.py --input_dir /path/to/obj_dir

Controls
--------
Mouse : rotate / zoom / pan (handled by PyVista)
→      : next timestep
←      : previous timestep
q / Esc: quit
"""

import argparse, re
from pathlib import Path

import numpy as np
import trimesh
import pyvista as pv


# ─────────────────────────────── helpers ──────────────────────────────
def extract_index(path: Path) -> int:
    """Grab the first group of digits in a filename for numeric sort."""
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def load_vertices(obj_path: Path) -> np.ndarray:
    """Load OBJ as a point cloud and return (N,3) array of vertices."""
    mesh = trimesh.load(obj_path, process=False)
    return np.asarray(mesh.vertices, dtype=np.float32)


def build_glyph(points: np.ndarray, color: str, radius: float = 0.002) -> pv.PolyData:
    """Turn raw points into a rendered sphere-glyph cloud."""
    cloud = pv.PolyData(points)
    # Use one sphere glyph per point (GPU instancing)
    return cloud.glyph(scale=False,
                       geom=pv.Sphere(radius=radius),
                       color=color)


def collect_pairs(input_dir: Path):
    """
    Return two aligned lists:
      regular_objs[i]   – nth frame's strands OBJ
      guide_objs[i]     – nth frame's guide OBJ (may be None)
    """
    regular = sorted([p for p in input_dir.glob("*.obj")
                      if not p.name.startswith("guide_")],
                     key=extract_index)
    guide = {extract_index(p): p for p in input_dir.glob("guide_*.obj")}

    regular_objs, guide_objs = [], []
    for reg in regular:
        idx = extract_index(reg)
        regular_objs.append(reg)
        guide_objs.append(guide.get(idx))            # may be None
    return regular_objs, guide_objs


# ─────────────────────────── main viewer ──────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True,
                    help="Directory containing OBJ strands (and optional guide_*.obj)")
    ap.add_argument("--radius", type=float, default=0.002,
                    help="Sphere radius for glyphs (in scene units)")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"{input_dir} is not a directory")

    regular_paths, guide_paths = collect_pairs(input_dir)
    if not regular_paths:
        raise RuntimeError("No strand OBJ files found!")

    print(f"Found {len(regular_paths)} timesteps "
          f"({len([g for g in guide_paths if g])} have guides).")
    print("Pre-loading geometries …")

    # Pre-load & glyph every frame so switching is instant
    regular_clouds = []
    guide_clouds   = []
    for reg_p, guide_p in zip(regular_paths, guide_paths):
        reg_pts   = load_vertices(reg_p)
        reg_glyph = build_glyph(reg_pts, "black", args.radius)
        regular_clouds.append(reg_glyph)

        if guide_p is not None:
            guide_pts   = load_vertices(guide_p)
            guide_glyph = build_glyph(guide_pts, "red", args.radius)
        else:
            guide_glyph = None
        guide_clouds.append(guide_glyph)

    # ── PyVista interactive scene ─────────────────────────────────────
    pv.set_plot_theme("document")
    pl = pv.Plotter(window_size=(900, 900))
    frame_idx = {"val": 0}             # use dict for mutability inside callbacks

    # Add initial meshes
    reg_actor  = pl.add_mesh(regular_clouds[0])
    guide_actor = (pl.add_mesh(guide_clouds[0])
                   if guide_clouds[0] is not None else None)

    pl.add_text(f"Frame 0 / {len(regular_clouds)-1}", name="frame_label",
                position="upper_left", font_size=14)

    # ── update callback ───────────────────────────────────────────────
    def show_frame(i: int):
        i %= len(regular_clouds)
        frame_idx["val"] = i

        reg_actor.mapper.SetInputData(regular_clouds[i])
        if guide_clouds[i] is not None:
            if guide_actor is None:
                # first time we encounter a guide cloud
                nonlocal guide_actor
                guide_actor = pl.add_mesh(guide_clouds[i])
            else:
                guide_actor.mapper.SetInputData(guide_clouds[i])
            guide_actor.SetVisibility(True)
        elif guide_actor is not None:
            guide_actor.SetVisibility(False)

        pl.update_text(f"Frame {i} / {len(regular_clouds)-1}", name="frame_label")
        pl.render()

    # ── key bindings ──────────────────────────────────────────────────
    def next_frame():
        show_frame(frame_idx["val"] + 1)

    def prev_frame():
        show_frame(frame_idx["val"] - 1)

    pl.add_key_event("Right", next_frame)
    pl.add_key_event("Left",  prev_frame)

    # nice initial view (hair roughly fits in [-0.1,0.1]^3)
    pl.set_background("white")
    pl.camera.position = (0.25, 0.0, 0.15)
    pl.camera.focal_point = (0.0, 0.0, 0.0)
    pl.camera.up = (0.0, 0.0, 1.0)

    print("Viewer controls:\n  • Mouse to rotate / zoom / pan\n"
          "  • → / ← to step frames\n  • q or Esc to quit")

    pl.show(title="Interactive Strand Viewer")


if __name__ == "__main__":
    main()