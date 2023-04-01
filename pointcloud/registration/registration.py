import open3d as o3d
import argparse
import numpy as np
import copy

def registration(source_pcd:o3d.geometry.PointCloud, target_pcd:o3d.geometry.PointCloud)->o3d.geometry.PointCloud:
    obj_func = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    result = o3d.pipelines.registration.registration_icp(source_pcd, target_pcd, 0.05, np.identity(4), obj_func)
    return  copy.deepcopy(source_pcd).transform(result.transformation)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_pcd_path",
        type=str,
        help="source pcd file path",
    )
    parser.add_argument(
        "target_pcd_path",
        type=str,
        help="target pcd file path",
    )
    args = parser.parse_args()

    source_pcd = o3d.io.read_point_cloud(args.source_pcd_path)
    target_pcd = o3d.io.read_point_cloud(args.target_pcd_path)
    source_pcd_transformed = registration(source_pcd, target_pcd)

    target_pcd.paint_uniform_color([0, 1, 1])
    source_pcd_transformed.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([source_pcd_transformed, target_pcd])