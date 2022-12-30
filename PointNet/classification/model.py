import torch
import torch.nn as nn

from PointNet.models.feature_tnet import FeatureTNet
from PointNet.models.input_tnet import InputTNet


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
            nn.BatchNorm1d(),
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.BatchNorm1d(),
        )^^^^^^^^^
        x = bach_data.pos

        return
