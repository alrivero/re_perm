# visualize_strands_rotation.py

import argparse, os
from pathlib import Path
from glob import glob
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm

def render_strand_obj(obj_path, azim=0, elev=20):
    cloud = trimesh.load(obj_path, process=False)
    points = cloud.vertices

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               s=0.15, c='black', alpha=0.9)

    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-0.1, 0.1)
    ax.set_zlim(-0.15, 0.05)
    ax.axis('off')

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image

def collect_obj_files(input_dir):
    input_dir = Path(input_dir)
    guide_objs = sorted(glob(str(input_dir / "guide_*.obj")))
    regular_objs = sorted([f for f in glob(str(input_dir / "*.obj")) if "guide_" not in f])
    return guide_objs, regular_objs

def save_rotation_video(obj_paths, azim, elev, out_path):
    frames = []
    for obj_path in tqdm(obj_paths, desc=f"Rendering azim={azim}° → {out_path}"):
        frame = render_strand_obj(obj_path, azim=azim, elev=elev)
        frames.append(frame)

    imageio.mimsave(out_path, frames, fps=10)
    print(f"[✓] Saved video: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with OBJ strands')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for videos')
    parser.add_argument('--step', type=int, default=30, help='Azimuth step in degrees')
    parser.add_argument('--elev', type=int, default=20, help='Elevation angle in degrees')
    args = parser.parse_args()

    guide_objs, regular_objs = collect_obj_files(args.input_dir)

    out_guide_dir = Path(args.out_dir) / "guide"
    out_regular_dir = Path(args.out_dir) / "regular"
    out_guide_dir.mkdir(parents=True, exist_ok=True)
    out_regular_dir.mkdir(parents=True, exist_ok=True)

    for azim in range(0, 360, args.step):
        guide_out = out_guide_dir / f"angle_{azim:03d}.mp4"
        reg_out   = out_regular_dir / f"angle_{azim:03d}.mp4"

        if guide_objs:
            save_rotation_video(guide_objs, azim, args.elev, guide_out)
        if regular_objs:
            save_rotation_video(regular_objs, azim, args.elev, reg_out)

if __name__ == "__main__":
    main()