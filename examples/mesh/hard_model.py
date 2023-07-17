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

        X_x, X_y = grad(X, [x, y])
        Y_x, Y_y = grad(X, [x, y])
        X_xy, X_xx = grad(X_x, [x, y])
        Y_yx, Y_yy = grad(Y_y, [x, y])
        X_yy = grad(X_y, y)[0]
        Y_xx = grad(Y_x, x)[0]
        return X, Y, X_xy, Y_yx, X_xx, X_yy, Y_xx, Y_yy
    
    def calc(self, Uin):
        x, y = Uin[:,0:1], Uin[:,1:2]
        U = self.linear_sigmoid_stack(Uin)
        X_nn, Y_nn = U[:,0:1], U[:,1:2]
        X = x + self.distance * X_nn
        Y = y + self.distance * Y_nn
        return torch.hstack((X, Y))