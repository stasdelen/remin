import torch
from torch import nn

class Wave(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(2, 32)
        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def forward(self, t, x):
        xs = torch.hstack((t, x))
        xs = torch.sin(self.input_layer(xs))
        return self.linear_sigmoid_stack(xs)