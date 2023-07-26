from remin.residual import Residual
from remin import func
from remin import domain as dm
import torch


def pde_residual(u_pred, t, x, y):
    u_t, u_x, u_y = func.grad(u_pred, [t, x, y])
    u_xx = func.grad(u_x, x)[0]
    u_yy = func.grad(u_y, y)[0]
    return u_t - (1/3) * (u_xx + u_yy),

def ic_residual(u_pred, t, x, y):
    return u_pred - x*y,

def bc0_residual(u_pred, t, x, y):
    return u_pred - torch.tensor(1.0),

def bc1_residual(u_pred, t, x, y):
    return u_pred - torch.tensor(0.5),

pde_col = dm.Time(0, 5) * dm.HyperCube([(0, 1), (0, 1)], 5000)
ic_col = dm.Time(0, 0) * dm.HyperCube([(0, 1), (0, 1)], 1000)
bc0_col = dm.Time(0, 5) * dm.Line((0, 0), (1, 0), 1000)
bc1_col = dm.Time(0, 5) * dm.Line((1, 0), (1, 1), 1000)
bc2_col = dm.Time(0, 5) * dm.Line((0, 1), (1, 1), 1000)
bc3_col = dm.Time(0, 5) * dm.Line((0, 0), (0, 1), 1000)

pde_res = Residual(pde_col, pde_residual)
ic_res = Residual(ic_col, ic_residual, weight=2)
bc0_res = Residual(bc0_col, bc0_residual, weight=1.5)
bc1_res = Residual(bc1_col, bc0_residual, weight=1.5)
bc2_res = Residual(bc2_col, bc1_residual, weight=1.5)
bc3_res = Residual(bc3_col, bc1_residual, weight=1.5)

residuals = [pde_res, ic_res, bc0_res, bc1_res, bc2_res, bc3_res]