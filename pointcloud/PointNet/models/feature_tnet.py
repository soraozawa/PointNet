import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_max_pool

from pointcloud.PointNet.data.load_ModelNet import load_ModelNet


class FeatureTNet(nn.Module):
    def __init__(self):
        super(FeatureTNet, self).__init__()
        self.input_mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64 * 64),
        )

    def forward(self, x, batch: torch.tensor):
        x = self.input_mlp(x)
        x = global_max_pool(x, batch)
        x = self.output_mlp(x)
        x = x.view(-1, 64, 64)
        id_matrix = (
            torch.eye(64).to(x.device).view(1, 64, 64).repeat(x.shape[0], 1, 1)
        )
        x = id_matrix + x
        return x
