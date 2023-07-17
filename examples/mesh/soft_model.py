import torch
from torch import nn
from remin.func import grad

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
        X, Y = U[:,0:1], U[:,1:2]
        X_x, X_y = grad(X, [x, y])
        Y_x, Y_y = grad(X, [x, y])
        X_xy, X_xx = grad(X_x, [x, y])
        Y_yx, Y_yy = grad(Y_y, [x, y])
        X_yy = grad(X_y, y)[0]
        Y_xx = grad(Y_x, x)[0]
        return X, Y, X_xy, Y_yx, X_xx, X_yy, Y_xx, Y_yy
    
    def calc(self, U):
        return self.linear_sigmoid_stack(U)