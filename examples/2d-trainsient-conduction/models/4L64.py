import torch
from torch import nn
from remin.solver.residual_loss import EagerLoss

class Conduction(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def forward(self, t, x, y):
        xs = torch.hstack((t, x, y))
        return self.linear_sigmoid_stack(xs)

residual_loss = EagerLoss(nn.MSELoss())
metric_loss = EagerLoss(nn.HuberLoss())