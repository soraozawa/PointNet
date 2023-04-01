import open3d as o3d
import numpy as np

def icp(source_pcd:o3d.geometry.PointCloud, target_pcd:o3d.geometry.PointCloud)->np.ndarray:
    transport_mat = np.eye(4)
    return transport_mat