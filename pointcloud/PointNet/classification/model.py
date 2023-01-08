import torch
import torch.nn as nn
from torch_geometric.nn import global_max_pool

from pointcloud.PointNet.models.feature_tnet import FeatureTNet
from pointcloud.PointNet.models.input_tnet import InputTNet


class PointNetClassification(nn.Module):
    def __init__(self):
        super(PointNetClassification, self).__init__()
        self.input_tnet = InputTNet()
        self.mlp1 = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.feature_tnet = FeatureTNet()
        self.mlp2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.BatchNorm1d(10),
        )

    def forward(self, batch_data):
        x = batch_data.pos

        input_transform = self.input_tnet(x, batch_data.batch)
        transform = input_transform[batch_data.batch, :, :]
        x = torch.bmm(transform, x.view(-1, 3, 1)).view(-1, 3)

        x = self.mlp1(x)

        feature_transform = self.feature_tnet(x, batch_data.batch)
        transform = feature_transform[batch_data.batch, :, :]
        x = torch.bmm(transform, x.view(-1, 64, 1)).view(-1, 64)

        x = self.mlp2(x)
        x = global_max_pool(x, batch_data.batch)
        x = self.mlp3(x)

        return x
