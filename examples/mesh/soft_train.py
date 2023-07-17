from remin.solver import Solver, make_trainer
from remin.residual import Residual, make_loader
from remin.solver.residual_loss import EagerLoss
from remin import callbacks
import torch
import numpy as np
from torch import nn
from soft_model import Mesh
from pre_process import createPoints


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
torch.set_float32_matmul_precision('high')
torch.manual_seed(0)
np.random.seed(0)

# Lame Parameters
mu_a = 0.35
lambda_a = 1.0

WALL_1, WALL_2, WALL_3, WALL_4, res_col, _, _, _, _ = createPoints('square.msh')

def pde_residual_X(grads, x, y):
    _, _, _, Y_yx, X_xx, X_yy, _, _= grads
    return lambda_a*(X_xx + Y_yx) + mu_a*(2*X_xx + X_yy + Y_yx)

def pde_residual_Y(grads, x, y):
    _, _, X_xy, _, _, _, Y_xx, Y_yy = grads
    return mu_a*(X_xy + Y_xx + 2*Y_yy) + lambda_a*(X_xy + Y_yy)

def stationary_wall_X(grads, x, y):
    X = grads[0]
    return X - x

def stationary_wall_Y(grads, x, y):
    Y = grads[1]
    return Y - y

def moving_wall_Y(grads, x, y):
    Y = grads[1]
    return Y - (y - 0.25 * torch.sin(torch.pi * x))

pde_res = Residual(res_col, [pde_residual_X, pde_residual_Y])
wall_1_res = Residual(WALL_1, [stationary_wall_X, stationary_wall_Y], weight=25)
wall_2_res = Residual(WALL_2, [stationary_wall_X, stationary_wall_Y], weight=25)
wall_3_res = Residual(WALL_3, [stationary_wall_X, stationary_wall_Y], weight=25)
wall_4_res = Residual(WALL_4, [stationary_wall_X, moving_wall_Y], weight=25)

if __name__ == '__main__':
    
    model = Mesh()

    loader = make_loader(
        [pde_res, wall_1_res, wall_2_res, wall_3_res, wall_4_res],
        fully_loaded=True
    )

    epochs = 10000
    lr = 5e-5

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    resloss = EagerLoss(nn.MSELoss())
    metloss = EagerLoss(nn.HuberLoss())
    trainer = make_trainer(loader,
                           optimizer=optimizer,
                           residual_loss=resloss,
                           metric_loss=metloss)
    
    solver = Solver(model,
                    name='mesh_square',
                    save_folder='./mesh_w25_lr5e5',
                    trainer=trainer)
    
    solver.reset_callbacks(
        callbacks.TotalTimeCallback(),
        callbacks.SaveCallback(),
        callbacks.CSVCallback(),
        callbacks.LogCallback(log_epoch=1000, log_progress=100),
        callbacks.PlotCallback(state='residual', name='ressloss.png'),
        callbacks.PlotCallback(state='metric', name='metloss.png')
    )

    solver.fit(epochs)