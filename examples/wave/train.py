from remin.solver import Solver, make_trainer
from remin.residual import Residual, make_loader
from remin.solver.residual_loss import EagerLoss
from remin import func, callbacks
from remin import domain as dm
import torch
import numpy as np
from torch import nn
from model import Wave

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
torch.set_float32_matmul_precision('high')
torch.manual_seed(0)
np.random.seed(0)

def pde_residual(u, t, x):
    u_t, u_x = func.grad(u, [t, x])
    u_tt = func.grad(u_t, t)[0]
    u_xx = func.grad(u_x, x)[0]
    return u_tt - u_xx

def ic0_residual(u, t, x):
    return func.grad(u, t)[0]

def ic1_residual(u, t, x):
    return u - torch.sin(3 * torch.pi * x)

def bc_residual(u, t, x):
    return u

pde_col = dm.Time(0, 1) * dm.Line((0,), (1,), 1024)
ic_col =  dm.Time(0, 0) * dm.Line((0,), (1,), 1024)
bc_col = dm.Time(0, 1) * (
    dm.Point((0,), 1024) + dm.Point((1,), 1024)
    )

pde_res = Residual(pde_col, pde_residual)
ic_res = Residual(ic_col, [ic0_residual, ic1_residual])
bc_res = Residual(bc_col, bc_residual)

if __name__ == '__main__':
    
    model = Wave()

    loader = make_loader(
        [pde_res, ic_res, bc_res],
        fully_loaded=True
    )

    epochs = 20000
    lr = 5e-4

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    resloss = EagerLoss(nn.MSELoss())
    metloss = EagerLoss(nn.HuberLoss())
    trainer = make_trainer(loader,
                           optimizer=optimizer,
                           residual_loss=resloss,
                           metric_loss=metloss)
    
    solver = Solver(model,
                    'wave_eager',
                    'outputs',
                    trainer=trainer)
    
    solver.reset_callbacks(
        callbacks.TotalTimeCallback(),
        callbacks.SaveCallback(),
        callbacks.LogCallback(log_epoch=1000, log_progress=100),
        callbacks.PlotCallback(state='resloss', name='eager_ressloss.png'),
        callbacks.PlotCallback(state='metloss', name='eager_metloss.png')
    )

    solver.fit(epochs)
