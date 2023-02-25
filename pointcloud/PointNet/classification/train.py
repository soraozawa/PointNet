from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pointcloud.PointNet.classification.model import PointNetClassification
from pointcloud.PointNet.data.load_ModelNet import load_ModelNet

num_epoch = 400
batch_size = 32

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

model = PointNetClassification()

model.to(device)

optimizer = torch.optim.Adam(lr=1e-4, params=model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=num_epoch // 4, gamma=0.5
)

log_dir = Path("data/log_modelnet10_classification")
log_dir.mkdir(exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

train_dataset, test_dataset = load_ModelNet(Path("data/modelnet10/modelnet10"))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
