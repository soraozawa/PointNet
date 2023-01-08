import torch
from torch.utils.tensorboard import SummaryWriter

from pointcloud.PointNet.classification.model import PointNetClassification

num_epoch = 400
batch_size = 32

device = (
    torch.device("cuda:0")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

model = PointNetClassification()

model.to(device)

optimizer = torch.optim.Adam(lr=1e-4, params=model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=num_epoch // 4, gamma=0.5
)

# log_di
