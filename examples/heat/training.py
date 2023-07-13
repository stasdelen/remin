from remin.solver import Solver, make_trainer
from remin.residual import Residual, make_loader
from remin.solver.residual_loss import FuncLoss
from remin import func, domain, callbacks
import torch
import numpy as np
from torch import nn
from model import Heat

torch.set_default_device('cuda')
torch.set_float32_matmul_precision('high')
    
model = Heat()

u = func.functional_call(model)
u_t = func.fgrad(u, argnum=1)
u_x = func.fgrad(u, argnum=2)
u_xx = func.fgrad(u_x, argnum=2)

def pde_residual(params, t, x):
    return u_t(params, t, x) - 0.05*u_xx(params, t, x)

def ic_residual(params, t, x):
    return u(params, t, x) - torch.tensor(1.0)

def bc_residual(params, t, x):
    return u(params, t, x) - torch.tensor(0.0)

pde_col = domain.HyperCube([(0, 1), (0, 1)], 1000)
ic_col = domain.Line((0, 0), (0, 1), 1000)
bc0_col = domain.Line((0, 0), (1, 0), 1000)
bc1_col = domain.Line((0, 1), (1, 1), 1000)

pde_res = Residual(pde_col, pde_residual)
ic_res = Residual(ic_col, ic_residual)
bc0_res = Residual(bc0_col, bc_residual)
bc1_res = Residual(bc1_col, bc_residual)

if __name__ == '__main__':
    loader = make_loader(
        [pde_res, ic_res, bc0_res, bc1_res],
        fully_loaded=True,
        #batched=False,
        #pin_memory=True,
        #num_workers=1
    )

    epochs = 35000
    lr = 2e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    resloss = FuncLoss(nn.MSELoss())
    metloss = FuncLoss(nn.HuberLoss())
    trainer = make_trainer(loader,
                       optimizer=optimizer,
                       residual_loss=resloss,
                       metric_loss=metloss)
    heat = Solver(model,
                  'heat',
                  'outputs',
                  trainer=trainer)
    
    heat.reset_callbacks(
        callbacks.SaveCallback(),
        callbacks.LogCallback(log_epoch=1000, log_progress=100),
        callbacks.PlotCallback(state='residual', name='resloss.png'),
        callbacks.PlotCallback(state='metric', name='metloss.png')
    )
    
    heat.fit(epochs)