import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_max_pool

from PointNet.data.load_ModelNet import load_ModelNet


class FeatureTNet(nn.Module):
    def __init__(self):
        super(FeatureTNet, self).__init__()
        # TODO
