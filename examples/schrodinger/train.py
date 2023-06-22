from remin.solver import Solver, make_trainer
from remin.residual import Residual, make_loader
from remin.solver.residual_loss import EagerLoss
from remin import func, domain, callbacks
import torch
import numpy as np
from torch import nn
from model import Shrodinger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
torch.set_float32_matmul_precision('high')
torch.manual_seed(0)
np.random.seed(0)

omega = 0.5
E = 2.75

def de_residual(u_pred, x):
    V = 0.5 * omega**2 * x**2
    du = func.grad(u_pred, x)[0]
    ddu = func.grad(du, x)[0]
    return -0.5*ddu + (V - E) * u_pred

def ic1_residual(u_pred, x):
    return u_pred

def ic2_residual(u_pred, x):
    return func.grad(u_pred, x)[0] - 0.86

de_col = domain.Line((-10,), (10,), 100)
ic_col = domain.Point((0,))

de_res = Residual(de_col, de_residual)
ic_res = Residual(ic_col, [ic1_residual, ic2_residual])

if __name__ == '__main__':
    
    model = Shrodinger()

    loader = make_loader(
        [de_res, ic_res],
        fully_loaded=True
    )

    epochs = 20000
    lr = 5e-3

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    resloss = EagerLoss(nn.MSELoss())
    metloss = EagerLoss(nn.HuberLoss())
    trainer = make_trainer(loader,
                           optimizer=optimizer,
                           residual_loss=resloss,
                           metric_loss=metloss)
    
    solver = Solver(model,
                    'schrodinger_eager',
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
