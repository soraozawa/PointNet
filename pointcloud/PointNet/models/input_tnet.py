import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_max_pool

from pointcloud.PointNet.data.load_ModelNet import load_ModelNet


class InputTNet(nn.Module):
    def __init__(self):
        super(InputTNet, self).__init__()
        self.input_mlp = nn.Sequential(
            nn.Linear(3, 64),
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
            nn.Linear(256, 9),
            nn.BatchNorm1d(9),
            nn.ReLU(),
        )

    def forward(self, x, batch):
        x = self.input_mlp(x)
        x = global_max_pool(x, batch)
        x = self.output_mlp(x)
        x = x.view(-1, 3, 3)
        id_matrix = torch.eye(3).to(x.device).view(1, 3, 3).repeat(x.shape[0], 1, 1)
        x = id_matrix + x
        return x
