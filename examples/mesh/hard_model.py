import torch
from torch import nn
from remin.func import grad


class MeshHardBC(nn.Module):
    def __init__(self, distance):
        super().__init__()
        # Distance function
        self.distance = distance
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
        Upar = torch.hstack((x, y))
        Uhat = self.linear_sigmoid_stack(Upar)
        X_nn, Y_nn = Uhat[:,0:1], Uhat[:,1:2]
        X = x + self.distance * X_nn
        Y = y + self.distance * Y_nn
        
        return X, Y
    
    def calc(self, Uin):
        x, y = Uin[:,0:1], Uin[:,1:2]
        U = self.linear_sigmoid_stack(Uin)
        X_nn, Y_nn = U[:,0:1], U[:,1:2]
        X = x + self.distance * X_nn
        Y = y + self.distance * Y_nn
        return torch.hstack((X, Y))