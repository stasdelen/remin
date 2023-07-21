from remin.solver import Solver, make_trainer
from remin.residual import Residual, make_loader
from remin.solver.residual_loss import EagerLoss
from remin.callbacks import LogCallback, PlotCallback, SaveCallback
from remin import domain, func
import torch
from torch import nn
from model import Heat

torch.set_default_device('cuda')
torch.set_float32_matmul_precision('high')


def pde_residual(u_pred, t, x):
    u_t, u_x = func.grad(u_pred, [t, x])
    u_xx = func.grad(u_x, x)[0]
    return u_t - 0.05*u_xx,

def ic_residual(u_pred, t, x):
    return u_pred - torch.tensor(1.0),

def bc_residual(u_pred, t, x):
    return u_pred - torch.tensor(0.0),

pde_col = domain.HyperCube([(0, 1), (0, 1)], 1000)
ic_col = domain.Line((0, 0), (0, 1), 1000)
bc0_col = domain.Line((0, 0), (1, 0), 1000)
bc1_col = domain.Line((0, 1), (1, 1), 1000)

pde_res = Residual(pde_col, pde_residual)
ic_res = Residual(ic_col, ic_residual, weight=2)
bc0_res = Residual(bc0_col, bc_residual, weight=1.5)
bc1_res = Residual(bc1_col, bc_residual, weight=1.5)

if __name__ == '__main__':
    
    model = Heat()    

    loader = make_loader(
        [pde_res, ic_res, bc0_res, bc1_res],
        fully_loaded=True
    )

    epochs = 10000
    lr = 8e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    resloss = EagerLoss(nn.HuberLoss())
    metloss = EagerLoss(nn.MSELoss())
    trainer = make_trainer(loader,
                           optimizer=optimizer,
                           residual_loss=resloss,
                           metric_loss=metloss)
    heat = Solver(model,
                  'heat',
                  'outputs',
                  trainer=trainer)
    
    heat.reset_callbacks(
        SaveCallback(),
        LogCallback(log_epoch=1000, log_progress=100),
        PlotCallback(state='residual', name='resloss.png'),
        PlotCallback(state='metric', name='metloss.png')
    )
    
    heat.fit(epochs)