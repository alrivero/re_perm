import numpy as np
import trimesh
import open3d as o3d
import argparse

def load_mesh_as_points(filename):
    mesh = trimesh.load(filename, process=False)
    return np.array(mesh.vertices), mesh

def prepare_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def run_icp(source_points, target_points, threshold=0.01, max_iteration=50):
    source_pcd = prepare_point_cloud(source_points)
    target_pcd = prepare_point_cloud(target_points)

    icp_result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    )
    return icp_result.transformation

def main(file_a, file_b, scale_factor, output_file):
    # Load points and meshes
    points_a, mesh_a = load_mesh_as_points(file_a)
    points_b, _ = load_mesh_as_points(file_b)

    # Apply scale for ICP only
    points_a_scaled = points_a / scale_factor

    # Run ICP on scaled points
    transformation_icp = run_icp(points_a_scaled, points_b)

    # Incorporate scaling into the transform
    final_transformation = transformation_icp.copy()
    final_transformation[:3, :3] /= scale_factor  # Divide rotation part by scale
    # translation stays the same

    print("Final transformation matrix (incorporating division scale):")
    print(final_transformation)

    # Apply final transform directly to original (unscaled) points
    points_a_transformed = (points_a @ final_transformation[:3, :3].T) + final_transformation[:3, 3]

    # Save transformed mesh or point cloud
    if hasattr(mesh_a, 'faces') and mesh_a.faces is not None and len(mesh_a.faces) > 0:
        transformed_mesh = trimesh.Trimesh(vertices=points_a_transformed, faces=mesh_a.faces, process=False)
    else:
        transformed_mesh = trimesh.PointCloud(points_a_transformed)

    transformed_mesh.export(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ICP between two OBJ files with scaling incorporated into the final transform.")
    parser.add_argument("--file_a", type=str, required=True, help="Path to source OBJ file (File A)")
    parser.add_argument("--file_b", type=str, required=True, help="Path to target OBJ file (File B)")
    parser.add_argument("--scale", type=float, required=True, help="Scale factor to divide points in File A before ICP")
    parser.add_argument("--output", type=str, required=True, help="Output path for transformed A OBJ")

    args = parser.parse_args()
    main(args.file_a, args.file_b, args.scale, args.output)