from remin.solver import Solver, make_trainer
from remin.residual import Residual, make_loader
from remin.solver.residual_loss import EagerLoss
from remin import domain, callbacks
from remin.func import grad
import torch
import numpy as np
from torch import nn
from taylorGreen_model import BNS
from taylorGreen_config import *
import sys
import matplotlib.pyplot as plt

torch.set_default_device('cuda')
torch.set_float32_matmul_precision('high')
torch.manual_seed(23)
np.random.seed(23)

print('----------------------------------------')
print('BNS Taylor Green Flow RT: ', str(RT), ' tau: ', str(tau))
print('----------------------------------------')

SQRT2 = np.sqrt(2)
# PI = np.pi

SCALE = 1e4

def exact(t, x, y):
	u = -torch.cos(x)*torch.sin(y)*torch.exp(-2*t*nu)
	v = torch.sin(x)*torch.cos(y)*torch.exp(-2*t*nu)

	return u, v

def neq_boundary(t, x, y):
	q4_t = 4*nu/RT * torch.cos(x)*torch.sin(x)*torch.cos(y)*torch.sin(y)*torch.exp(-4*nu*t)
	q5_t = -4*nu/(SQRT2*RT) * torch.cos(x)**2 * torch.sin(y)**2 * torch.exp(-4*nu*t)
	q6_t = -4*nu/(SQRT2*RT) * torch.sin(x)**2 * torch.cos(y)**2 * torch.exp(-4*nu*t)

	q2_x =  1/sqrtRT * torch.sin(x)*torch.sin(y)*torch.exp(-2*nu*t)
	q2_y = -1/sqrtRT * torch.cos(x)*torch.cos(y)*torch.exp(-2*nu*t)

	q3_x =  1/sqrtRT * torch.cos(x)*torch.cos(y)*torch.exp(-2*nu*t)
	q3_y = -1/sqrtRT * torch.sin(x)*torch.sin(y)*torch.exp(-2*nu*t)

	neq_4 = -tau * (q4_t + sqrtRT * (q3_x + q2_y))
	neq_5 = -tau * (q5_t + sqrtRT * (SQRT2*q2_x))
	neq_6 = -tau * (q6_t + sqrtRT * (SQRT2*q3_y))

	return neq_4, neq_5, neq_6

def pde_residual(U, t, x, y):
	rho, u, v, n1, n2, n3 = U

	q1_eq = rho
	q2_eq = rho * u / sqrtRT
	q3_eq = rho * v / sqrtRT
	q4_eq = rho * u * v / RT
	q5_eq = rho * u * u / (SQRT2*RT)
	q6_eq = rho * v * v / (SQRT2*RT)

	q1 = q1_eq
	q2 = q2_eq
	q3 = q3_eq
	q4 = q4_eq + n1/SCALE
	q5 = q5_eq + n2/SCALE
	q6 = q6_eq + n3/SCALE

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

	f1 = q1_t + sqrtRT * (q2_x + q3_y)
	f2 = q2_t + sqrtRT * (q1_x + q4_y + SQRT2*q5_x)
	f3 = q3_t + sqrtRT * (q4_x + q1_y + SQRT2*q6_y)

	f4 = q4_t + sqrtRT * (q3_x + q2_y) + 1./tau * (q4 - q4_eq)
	f5 = q5_t + sqrtRT * (SQRT2*q2_x)  + 1./tau * (q5 - q5_eq)
	f6 = q6_t + sqrtRT * (SQRT2*q3_y)  + 1./tau * (q6 - q6_eq)

	return f1, f2, f3, f4, f5, f6

def dirichlet(U, t, x, y):
	rho, u, v, n1, n2, n3 = U

	# q1_eq = rho
	# q2_eq = rho * u / sqrtRT
	# q3_eq = rho * v / sqrtRT
	# q4_eq = rho * u * v / RT
	# q5_eq = rho * u * u / (SQRT2*RT)
	# q6_eq = rho * v * v / (SQRT2*RT)

	u_exact, v_exact = exact(t, x, y)

	f4_neq, f5_neq, f6_neq = neq_boundary(t, x, y)
	
	f4_neq = SCALE * f4_neq
	f5_neq = SCALE * f5_neq
	f6_neq = SCALE * f6_neq

	return rho-1, u-u_exact, v-v_exact, n1-f4_neq, n2-f5_neq, n3-f6_neq

def initial(U, t, x, y):
	rho, u, v, n1, n2, n3 = U

	u_exact, v_exact = exact(t, x, y)

	return rho-1, u-u_exact, v-v_exact


# Define the pde domain
pde_domain = domain.Time(tstart, tfinal) * domain.HyperCube([(xmin, xmax), (ymin, ymax)], n_coll=20000)
pde_col    = pde_domain
pde_points = pde_col.generate().get()

# Define the wall boundaries
wall_1_col = domain.Time(tstart, tfinal) * domain.Line((xmin,ymin), (xmax,ymin), n_coll=300)
wall_2_col = domain.Time(tstart, tfinal) * domain.Line((xmax,ymin), (xmax,ymax), n_coll=300)
wall_3_col = domain.Time(tstart, tfinal) * domain.Line((xmax,ymax), (xmin,ymax), n_coll=300)
wall_4_col = domain.Time(tstart, tfinal) * domain.Line((xmin,ymax), (xmin,ymin), n_coll=300)

wall_1_points = wall_1_col.generate().get()
wall_2_points = wall_2_col.generate().get()
wall_3_points = wall_3_col.generate().get()
wall_4_points = wall_4_col.generate().get()

# Define the initial domain
ic_domain   = domain.Time(tstart, tstart) * domain.HyperCube([(xmin, xmax), (ymin, ymax)], n_coll=1000)
ic_col      = ic_domain
ic_points   = ic_col.generate().get()


# plt.scatter(wall_1_points[:,0:1], wall_1_points[:,1:2])
# plt.scatter(wall_2_points[:,0:1], wall_2_points[:,1:2])
# plt.scatter(wall_3_points[:,0:1], wall_3_points[:,1:2])
# plt.scatter(wall_4_points[:,0:1], wall_4_points[:,1:2])

# plt.scatter(pde_points[:,0:1], pde_points[:,1:2])

# plt.show()
# sys.exit(1)

pde_res = Residual(pde_points, pde_residual)
ic_res  = Residual(ic_points, initial)

wall_1_res = Residual(wall_1_points, dirichlet)
wall_2_res = Residual(wall_2_points, dirichlet)
wall_3_res = Residual(wall_3_points, dirichlet)
wall_4_res = Residual(wall_4_points, dirichlet)

wall_1_pde = Residual(wall_1_points, pde_residual)
wall_2_pde = Residual(wall_2_points, pde_residual)
wall_3_pde = Residual(wall_3_points, pde_residual)
wall_4_pde = Residual(wall_4_points, pde_residual)

if __name__ == '__main__':

	model = BNS()

	loader = make_loader(
		[pde_res, ic_res, wall_1_res, wall_2_res, wall_3_res, wall_4_res,
		wall_1_pde, wall_2_pde, wall_3_pde, wall_4_pde],
		fully_loaded=True, batched=False)

	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=gamma)
	resloss = EagerLoss(nn.MSELoss())
	trainer = make_trainer(loader,
	                       optimizer=optimizer,
	                       scheduler=scheduler,
	                       residual_loss=resloss)

	solver = Solver(model,
	                name='bns',
	                save_folder=save_folder,
	                trainer=trainer)

	solver.reset_callbacks(
	    callbacks.TotalTimeCallback(),
	    callbacks.SaveCallback(),
	    callbacks.CSVCallback(),
	    callbacks.LogCallback(log_epoch=1000, log_progress=100),
	    callbacks.PlotCallback(state='residual', name='ressloss.png')
	)

	solver.fit(epochs)