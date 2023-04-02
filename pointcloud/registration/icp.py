import numpy as np
import open3d as o3d


def calc_covariance_matrix(pcd1: np.ndarray, pcd2: np.ndarray)->np.ndarray:
    """2つの点群の共分散行列を求める

    Args:
        pcd1 (np.ndarray): 点群1
        pcd2 (np.ndarray): 点群2

    Returns:
        np.ndarray: 共分散行列
    """
    source_pcd_mean = pcd1.mean(axis=0)
    neighbor_pcd_mean = pcd2.mean(axis=0)
    covar_mat = np.zeros((3,3))
    for source_point, neighbor_point in zip(pcd1, pcd2):
        covar_mat += np.dot(source_point.reshape(-1, 1), neighbor_point.reshape(1, -1))
    covar_mat /= len(pcd1)
    covar_mat -= np.dot(source_pcd_mean.reshape(-1, 1), neighbor_pcd_mean.reshape(1, -1))

    return covar_mat

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


    covar_mat = calc_covariance_matrix(np_source_pcd, np_neighbor_pcd)
    print(covar_mat)


    return transport_mat