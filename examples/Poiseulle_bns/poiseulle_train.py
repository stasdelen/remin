from remin.solver import Solver, make_trainer
from remin.residual import Residual, make_loader
from remin.solver.residual_loss import EagerLoss
from remin import domain, callbacks
from remin.func import grad
import torch
import numpy as np
from torch import nn
from poiseulle_model import BNS
import sys
import matplotlib.pyplot as plt

torch.set_default_device('cuda')
torch.set_float32_matmul_precision('high')
torch.manual_seed(23)
np.random.seed(23)

# Flow Parameters
uR = 1.0
c  = 10.0
Ma = uR / c
RT = c*c
L  = 1
nu = 0.001
Re = uR * L / nu
tau = nu / (c*c)

print('----------------------------------------')
print('BNS Poiseulle Re: ', str(Re), ' tau: ', str(tau))
print('----------------------------------------')

def pde_residual(U, t, x, y):
	q1, q2, q3, q4, q5, q6, n1, n2, n3 = U

	q1[q1<0.1] = 0.1

	q1_t = grad(q1, t)[0]
	q1_x = grad(q1, x)[0]
	q1_y = grad(q1, y)[0]

	q2_t = grad(q2, t)[0]
	q2_x = grad(q2, x)[0]
	q2_y = grad(q2, y)[0]

	q3_t = grad(q3, t)[0]
	q3_x = grad(q3, x)[0]
	q3_y = grad(q3, y)[0]

	q4_t = grad(q4, t)[0]
	q4_x = grad(q4, x)[0]
	q4_y = grad(q4, y)[0]

	q5_t = grad(q5, t)[0]
	q5_x = grad(q5, x)[0]
	# q5_y = grad(q5, y)[0]

	q6_t = grad(q6, t)[0]
	# q6_x = grad(q6, x)[0]
	q6_y = grad(q6, y)[0]

	f1 = q1_t + c * (q2_x + q3_y)
	f2 = q2_t + c * (q1_x + q4_y + np.sqrt(2)*q5_x)
	f3 = q3_t + c * (q4_x + q1_y + np.sqrt(2)*q6_y)

	f4 = q4_t + c * (q3_x + q2_y)     + 1/tau * (n1)
	f5 = q5_t + c * (np.sqrt(2)*q2_x) + 1/tau * (n2)
	f6 = q6_t + c * (np.sqrt(2)*q3_y) + 1/tau * (n3)

	# f4 = q4_t + c * (q3_x + q2_y)     + 1/tau * (q4 - q2*q3/q1)
	# f5 = q5_t + c * (np.sqrt(2)*q2_x) + 1/tau * (q5 - q2*q2/(np.sqrt(2)*q1))
	# f6 = q6_t + c * (np.sqrt(2)*q3_y) + 1/tau * (q6 - q3*q3/(np.sqrt(2)*q1))

	return f1, f2, f3, f4, f5, f6

def wall(U, t, x, y):
	q1, q2, q3, q4, q5, q6, n1, n2, n3 = U

	q1[q1<0.1] = 0.1
	rho = q1
	u   = q2 * c / rho
	v   = q3 * c / rho
	s11 = -RT * (np.sqrt(2)*q5 - q2*q2/q1)
	s22 = -RT * (np.sqrt(2)*q6 - q3*q3/q1)
	s12 = -RT * (q4 - q2*q3/q1)

	return rho-1, u, v, s11, s22, s12

def inflow(U, t, x, y):
	q1, q2, q3, q4, q5, q6, n1, n2, n3 = U

	q1[q1<0.1] = 0.1
	rho = q1
	u   = q2 * c / rho
	v   = q3 * c / rho

	u_in = (1 - y*y)*uR

	return rho-1, u-u_in, v

def outflow(U, t, x, y):
	q1, q2, q3, q4, q5, q6, n1, n2, n3 = U

	q1[q1<0.1] = 0.1
	rho = q1
	u   = q2 * c / rho
	v   = q3 * c / rho
	p   = rho*RT

	u_x = grad(u, x)[0]
	v_x = grad(v, x)[0]
	p_x = grad(p, x)[0]

	return u_x, v_x, p_x


def data(ruv):
	rho_d, u_d, v_d = ruv[:,0:1], ruv[:,1:2], ruv[:,2:3]
	rho_d = torch.from_numpy(rho_d).to('cuda')
	u_d   = torch.from_numpy(u_d).to('cuda')
	v_d   = torch.from_numpy(v_d).to('cuda')
	def data_res(U, t, x, y):
		q1, q2, q3, q4, q5, q6, n1, n2, n3 = U
		q1[q1<0.1] = 0.1
		rho = q1
		u = q2 * c / rho
		v = q3 * c / rho
		return rho_d-rho, u_d-u, v_d-v
	return data_res


def initial(U, t, x, y):
	q1, q2, q3, q4, q5, q6, n1, n2, n3 = U

	q1[q1<0.1] = 0.1

	rho = q1
	u   = q2 * c / rho
	v   = q3 * c / rho
	s11 = -RT * (np.sqrt(2)*q5 - q2*q2/q1)
	s22 = -RT * (np.sqrt(2)*q6 - q3*q3/q1)
	s12 = -RT * (q4 - q2*q3/q1)

	return rho-1, u, v, s11, s22, s12

xmin = 	0.0
xmax =  5.0
ymin = -1.0
ymax =  1.0

tstart = 0
tfinal = 5.0

pde_col    = domain.Time(tstart,tfinal) * domain.HyperCube([(xmin, xmax), (ymin, ymax)], n_coll=15000)
ic_col     = domain.Time(tstart,tstart) * domain.HyperCube([(xmin, xmax), (xmin, xmax)], n_coll=3500)
wall_1_col = domain.Time(tstart,tfinal) * domain.Line((xmin,ymin), (xmax,ymin), n_coll=3500)
wall_2_col = domain.Time(tstart,tfinal) * domain.Line((xmax,ymin), (xmax,ymax), n_coll=3500)
wall_3_col = domain.Time(tstart,tfinal) * domain.Line((xmax,ymax), (xmin,ymax), n_coll=3500)
wall_4_col = domain.Time(tstart,tfinal) * domain.Line((xmin,ymax), (xmin,ymin), n_coll=3500)

pde_res    = Residual(pde_col, pde_residual)
wall_1_res = Residual(wall_1_col, wall)
wall_2_res = Residual(wall_2_col, outflow)
wall_3_res = Residual(wall_3_col, wall)
wall_4_res = Residual(wall_4_col, inflow)
ic_res     = Residual(ic_col, initial)

if __name__ == '__main__':

	model = BNS(tau, c)

	loader = make_loader(
		[pde_res, wall_1_res, wall_2_res, wall_3_res, wall_4_res, ic_res],
		fully_loaded=True, batched=False)

	epochs = 30000
	lr = 1e-3
	gamma = 0.99

	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=gamma)
	resloss = EagerLoss(nn.MSELoss())
	trainer = make_trainer(loader,
	                       optimizer=optimizer,
	                       scheduler=scheduler,
	                       residual_loss=resloss)

	solver = Solver(model,
	                name='bns',
	                save_folder='./outputs/Poiseulle_Re1000_adaptive',
	                trainer=trainer)

	solver.reset_callbacks(
	    callbacks.TotalTimeCallback(),
	    callbacks.SaveCallback(),
	    callbacks.CSVCallback(),
	    callbacks.LogCallback(log_epoch=1000, log_progress=100),
	    callbacks.PlotCallback(state='residual', name='ressloss.png')
	)

	solver.fit(epochs)