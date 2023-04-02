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
    np_source_pcd = np.asarray(source_pcd.points)
    np_neighbor_pcd = np_target_pcd[idx_list].copy()
    print(idx_list)
    print(np_neighbor_pcd)

    source_pcd_mean = np_source_pcd.mean(axis=0)
    neighbor_pcd_mean = np_target_pcd.mean(axis=0)

    covar_mat = np.zeros((3,3))
    for source_point, neighbor_point in zip(np_source_pcd, np_neighbor_pcd):
        covar_mat += np.dot(source_point.reshape(-1, 1), neighbor_point.reshape(1, -1))
    covar_mat /= len(np_source_pcd)
    covar_mat -= np.dot(source_pcd_mean.reshape(-1, 1), neighbor_pcd_mean.reshape(1, -1))
    print(covar_mat)


    return transport_mat