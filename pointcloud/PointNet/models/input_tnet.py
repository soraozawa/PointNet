import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_max_pool

from pointcloud.PointNet.data.load_ModelNet import load_ModelNet


class InputTNet(nn.Module):
    def __init__(self):
        super(InputTNet, self).__init__()
        # TODO
