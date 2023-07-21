import torch
from torch import nn

class Mesh(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_sigmoid_stack = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 2)
        )
    
    def forward(self, x, y):
        Uin = torch.hstack((x, y))
        U = self.linear_sigmoid_stack(Uin)
        return U[:,0:1], U[:,1:2]
    
    def calc(self, U):
        return self.linear_sigmoid_stack(U)