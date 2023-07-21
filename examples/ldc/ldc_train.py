from remin.solver import Solver, make_trainer
from remin.residual import Residual, make_loader
from remin.solver.residual_loss import EagerLoss
from remin import domain, callbacks
from remin.func import grad
import torch
import numpy as np
from torch import nn
from ldc_model import Ldc

torch.set_default_device('cuda')
torch.set_float32_matmul_precision('high')
torch.manual_seed(23)
np.random.seed(23)

# Flow Parameters
Re  = 100.0
rho = 1.0
mu  = 1/Re

def pde_residual(U, x, y):
	u, v, p = U
	u_x = grad(u, x)[0]
	u_y = grad(u, y)[0]
	u_xx = grad(u_x, x)[0]
	u_yy = grad(u_y, y)[0]
	
	v_x = grad(v, x)[0]
	v_y = grad(v, y)[0]
	v_xx = grad(v_x, x)[0]
	v_yy = grad(v_y, y)[0]
	
	p_x = grad(p, x)[0]
	p_y = grad(p, y)[0]
	
	fx = rho*(u*u_x + v*u_y) + p_x - mu*(u_xx + u_yy)
	fy = rho*(u*v_x + v*v_y) + p_y - mu*(v_xx + v_yy)
	fc = u_x + v_y
	return fx, fy, fc

def no_slip(U, x, y):
	u, v = U[0], U[1]
	return u, v

def inflow_ldc(U, x, y):
	u, v = U[0], U[1]
	return u - 1, v


pde_col = domain.HyperCube([(0, 1), (0, 1)], n_coll=1000)
wall_1_col = domain.Line((0,0), (0,1), n_coll=120)
wall_2_col = domain.Line((1,0), (1,1), n_coll=120)
wall_3_col = domain.Line((0,0), (1,0), n_coll=120)
wall_4_col = domain.Line((0,1), (1,1), n_coll=120)
stationary_walls_col = wall_1_col + wall_2_col + wall_3_col 

pde_res = Residual(pde_col, pde_residual)
stationary_walls_res = Residual(stationary_walls_col, no_slip)
wall_4_res = Residual(wall_4_col, inflow_ldc)

if __name__ == '__main__':

	model = Ldc()

	loader = make_loader(
		[pde_res, stationary_walls_res, wall_4_res],
		fully_loaded=True)

	epochs = 15000
	lr = 1e-3
	gamma = 0.99

	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=gamma)
	resloss = EagerLoss(nn.MSELoss())
	trainer = make_trainer(loader,
	                       optimizer=optimizer,
	                       scheduler=scheduler,
	                       residual_loss=resloss)

	solver = Solver(model,
	                name='LDC_Re100',
	                save_folder='./outputs',
	                trainer=trainer)

	solver.reset_callbacks(
	    callbacks.TotalTimeCallback(),
	    callbacks.SaveCallback(),
	    callbacks.CSVCallback(),
	    callbacks.LogCallback(log_epoch=1000, log_progress=100),
	    callbacks.PlotCallback(state='residual', name='ressloss.png')
	)

	solver.fit(epochs)
