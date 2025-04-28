#!/usr/bin/env python3
"""
align_meshes.py

Scales the source mesh by the ratio of its AABB to the target's AABB,
then fits a rigid (rotation + translation) transform via ICP, and writes
out the final aligned mesh.
"""

import argparse
import numpy as np
import open3d as o3d

def parse_args():
    parser = argparse.ArgumentParser(
        description="Scale & align one OBJ to another using ICP."
    )
    parser.add_argument(
        "source", help="Path to the OBJ file to be aligned (the moving mesh)."
    )
    parser.add_argument(
        "target", help="Path to the OBJ file whose size/orientation you want to match."
    )
    parser.add_argument(
        "output", help="Output path for the aligned version of the source OBJ."
    )
    parser.add_argument(
        "--voxel_size", type=float, default=0.01,
        help="Voxel size for downsampling point clouds (default: 0.01)."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.02,
        help="Max correspondence point distance for ICP (default: 0.02)."
    )
    parser.add_argument(
        "--max_iter", type=int, default=50,
        help="Maximum number of ICP iterations (default: 50)."
    )
    parser.add_argument(
        "--num_points", type=int, default=200000,
        help="Number of points to sample uniformly from each mesh (default: 200k)."
    )
    return parser.parse_args()

def preprocess_point_cloud(pcd, voxel_size):
    """Downsample and estimate normals."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2, max_nn=30
        )
    )
    return pcd_down

def run_icp(source_down, target_down, threshold, max_iter):
    """Perform point-to-point ICP and return the transformation matrix."""
    icp_result = o3d.pipelines.registration.registration_icp(
        source_down, target_down,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )
    return icp_result.transformation, icp_result

def main():
    args = parse_args()

    # --- Load raw meshes ---
    source_mesh = o3d.io.read_triangle_mesh(args.source)
    target_mesh = o3d.io.read_triangle_mesh(args.target)

    # --- Compute and apply initial uniform scale by AABB ---
    bb_src = source_mesh.get_axis_aligned_bounding_box()
    bb_tgt = target_mesh.get_axis_aligned_bounding_box()
    size_src = bb_src.get_max_bound() - bb_src.get_min_bound()
    size_tgt = bb_tgt.get_max_bound() - bb_tgt.get_min_bound()
    scales = size_tgt / size_src
    init_scale = float(np.min(scales))
    source_mesh.scale(init_scale, center=bb_src.get_center())
    print(f"[Init] Applied uniform scale: {init_scale:.6f}")

    # --- Sample point clouds ---
    if not source_mesh.has_vertex_normals():
        source_mesh.compute_vertex_normals()
    source_pcd = source_mesh.sample_points_uniformly(number_of_points=args.num_points)

    if not target_mesh.has_vertex_normals():
        target_mesh.compute_vertex_normals()
    target_pcd = target_mesh.sample_points_uniformly(number_of_points=args.num_points)

    # --- Preprocess for ICP ---
    source_down = preprocess_point_cloud(source_pcd, args.voxel_size)
    target_down = preprocess_point_cloud(target_pcd, args.voxel_size)

    # --- Run ICP ---
    transformation, icp_info = run_icp(
        source_down, target_down,
        args.threshold, args.max_iter
    )

    # --- Apply the ICP transform to the (already–scaled) mesh ---
    source_mesh.transform(transformation)

    # --- Save the aligned mesh ---
    o3d.io.write_triangle_mesh(args.output, source_mesh)

    # --- Report results ---
    print("Final rigid transformation (4×4 matrix) applied after initial scale:")
    print(transformation)
    print(f"ICP fitness: {icp_info.fitness:.4f}, RMSE: {icp_info.inlier_rmse:.4f}")
    print(f"Aligned mesh written to: {args.output}")

if __name__ == "__main__":
    main()