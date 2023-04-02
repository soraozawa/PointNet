import numpy as np
import open3d as o3d


def icp(source_pcd:o3d.geometry.PointCloud, target_pcd:o3d.geometry.PointCloud)->np.ndarray:
    transport_mat = np.eye(4)

    pcd_tree = o3d.geometry.KDTreeFlann(target_pcd)
    idx_list = []
    for point in source_pcd.points:
        [k, idx, _] = pcd_tree.search_knn_vector_3d(point, 1)
        idx_list.append(idx[0])
    
    np_target_pcd = np.asarray(target_pcd.points)
    np_neighbor_pcd = np_target_pcd[idx_list].copy()
    print(idx_list)
    print(np_neighbor_pcd)
    return transport_mat