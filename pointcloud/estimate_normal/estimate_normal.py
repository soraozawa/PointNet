import argparse

import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument("pcd_path", type=str)
args = parser.parse_args()

pcd = o3d.io.read_point_cloud(args.pcd_path)

o3d.visualization.draw_geometries([pcd])


pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=10)
)
o3d.visualization.draw_geometries([pcd], point_show_normal=True)
