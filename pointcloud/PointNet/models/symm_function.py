import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_max_pool

from pointcloud.PointNet.data.load_ModelNet import load_ModelNet


class SymmFunction(nn.Module):
    def __init__(self):
        super(SymmFunction, self).__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 521),
        )

    def forward(self, batch: torch.tensor) -> torch.tensor:
        x = self.shared_mlp(batch.pos)
        x - global_max_pool(x)
        return x


if __name__ == "__main__":
    symm_func = SymmFunction()
    train_dataset, _ = load_ModelNet()
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    batch = next(iter(dataloader))
    print(batch)
    y = symm_func(batch)
    print(y.shape)
