import open3d as o3d
import numpy as np
import numpy.linalg as LA
import copy

pcd1 = o3d.io.read_point_cloud(".../data/bunny/bun000.ply")