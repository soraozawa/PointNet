
import copy

import numpy as np
import open3d as o3d

from pointcloud.registration.icp import icp


def test_icp():
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(np.array([[0,0,0], [1,1,1]]))
    target_pcd = copy.deepcopy(source_pcd)

    assert np.array_equal(icp(source_pcd, target_pcd), np.eye(4))