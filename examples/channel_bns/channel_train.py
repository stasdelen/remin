from remin.solver import Solver, make_trainer
from remin.residual import Residual, make_loader
from remin.solver.residual_loss import EagerLoss
from remin import domain, callbacks
from remin.func import grad
import torch
import numpy as np
from torch import nn
from channel_model import BNS
from channel_config import *
import sys
import matplotlib.pyplot as plt

torch.set_default_device('cuda')
torch.set_float32_matmul_precision('high')
torch.manual_seed(234)
np.random.seed(234)

print('----------------------------------------')
print('BNS Channel Re: ', str(Re), ' tau: ', str(tau))
print('----------------------------------------')

SQRT2 = np.sqrt(2)

SCALE = 1e4

def pde_residual(U, t, x, y):
	rho, u, v, n1, n2, n3 = U

	q1_eq = rho
	q2_eq = rho * u / c
	q3_eq = rho * v / c
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

	f1 = q1_t + c * (q2_x + q3_y)
	f2 = q2_t + c * (q1_x + q4_y + SQRT2*q5_x)
	f3 = q3_t + c * (q4_x + q1_y + SQRT2*q6_y)

	f4 = q4_t + c * (q3_x + q2_y) + 1./tau * (q4 - q4_eq)
	f5 = q5_t + c * (SQRT2*q2_x)  + 1./tau * (q5 - q5_eq)
	f6 = q6_t + c * (SQRT2*q3_y)  + 1./tau * (q6 - q6_eq)

	return f1, f2, f3, f4, f5, f6

def wall(U, t, x, y):
	rho, u, v, n1, n2, n3 = U

	q1_eq = rho
	q2_eq = rho * u / c
	q3_eq = rho * v / c
	q4_eq = rho * u * v / RT
	q5_eq = rho * u * u / (SQRT2*RT)
	q6_eq = rho * v * v / (SQRT2*RT)

	# q1_eq_t = grad(q1_eq, t)[0]
	# q1_eq_x = grad(q1_eq, x)[0]
	# q1_eq_y = grad(q1_eq, y)[0]

	# q2_eq_t = grad(q2_eq, t)[0]
	# q2_eq_x = grad(q2_eq, x)[0]
	# q2_eq_y = grad(q2_eq, y)[0]

	# # q3_eq_t = grad(q3_eq, t)[0]
	# q3_eq_x = grad(q3_eq, x)[0]
	# q3_eq_y = grad(q3_eq, y)[0]

	# q4_eq_t = grad(q4_eq, t)[0]
	# q4_eq_x = grad(q4_eq, x)[0]
	# q4_eq_y = grad(q4_eq, y)[0]

	# q5_eq_t = grad(q5_eq, t)[0]
	# # q5_eq_x = grad(q5_eq, x)[0]
	# # q5_eq_y = grad(q5_eq, y)[0]

	# q6_eq_t = grad(q6_eq, t)[0]
	# # q6_eq_x = grad(q6_eq, x)[0]
	# # q6_eq_y = grad(q6_eq, y)[0]

	# f4_neq = -tau * (q4_eq_t + c * (q3_eq_x + q2_eq_y))
	# f5_neq = -tau * (q5_eq_t + c * (SQRT2*q2_eq_x))
	# f6_neq = -tau * (q6_eq_t + c * (SQRT2*q3_eq_y))

	f4_neq = 0.0
	f5_neq = 0.0
	f6_neq = 0.0

	f4_neq = SCALE * f4_neq
	f5_neq = SCALE * f5_neq
	f6_neq = SCALE * f6_neq

	return rho-rBar, u, v, n1-f4_neq, n2-f5_neq, n3-f6_neq

def inflow(U, t, x, y):
	rho, u, v, n1, n2, n3 = U

	q1_eq = rho
	q2_eq = rho * u / c
	q3_eq = rho * v / c
	q4_eq = rho * u * v / RT
	q5_eq = rho * u * u / (SQRT2*RT)
	q6_eq = rho * v * v / (SQRT2*RT)

	# q1_eq_t = grad(q1_eq, t)[0]
	# q1_eq_x = grad(q1_eq, x)[0]
	# q1_eq_y = grad(q1_eq, y)[0]

	# q2_eq_t = grad(q2_eq, t)[0]
	# q2_eq_x = grad(q2_eq, x)[0]
	# q2_eq_y = grad(q2_eq, y)[0]

	# # q3_eq_t = grad(q3_eq, t)[0]
	# q3_eq_x = grad(q3_eq, x)[0]
	# q3_eq_y = grad(q3_eq, y)[0]

	# q4_eq_t = grad(q4_eq, t)[0]
	# q4_eq_x = grad(q4_eq, x)[0]
	# q4_eq_y = grad(q4_eq, y)[0]

	# q5_eq_t = grad(q5_eq, t)[0]
	# # q5_eq_x = grad(q5_eq, x)[0]
	# # q5_eq_y = grad(q5_eq, y)[0]

	# q6_eq_t = grad(q6_eq, t)[0]
	# # q6_eq_x = grad(q6_eq, x)[0]
	# # q6_eq_y = grad(q6_eq, y)[0]

	# f4_neq = -tau * (q4_eq_t + c * (q3_eq_x + q2_eq_y))
	# f5_neq = -tau * (q5_eq_t + c * (SQRT2*q2_eq_x))
	# f6_neq = -tau * (q6_eq_t + c * (SQRT2*q3_eq_y))

	f4_neq = 0.0
	f5_neq = 0.0
	f6_neq = 0.0

	f4_neq = SCALE * f4_neq
	f5_neq = SCALE * f5_neq
	f6_neq = SCALE * f6_neq

	r_in = rBar
	u_in = uBar
	v_in = vBar

	return rho-r_in, u-u_in, v-v_in, n1-f4_neq, n2-f5_neq, n3-f6_neq

def outflow(U, t, x, y):
	rho, u, v, n1, n2, n3 = U

	q1_eq = rho
	q2_eq = rho * u / c
	q3_eq = rho * v / c
	q4_eq = rho * u * v / RT
	q5_eq = rho * u * u / (SQRT2*RT)
	q6_eq = rho * v * v / (SQRT2*RT)

	# q1_eq_t = grad(q1_eq, t)[0]
	# q1_eq_x = grad(q1_eq, x)[0]
	# q1_eq_y = grad(q1_eq, y)[0]

	# q2_eq_t = grad(q2_eq, t)[0]
	q2_eq_x = grad(q2_eq, x)[0]
	q2_eq_y = grad(q2_eq, y)[0]

	# q3_eq_t = grad(q3_eq, t)[0]
	q3_eq_x = grad(q3_eq, x)[0]
	q3_eq_y = grad(q3_eq, y)[0]

	q4_eq_t = grad(q4_eq, t)[0]
	q4_eq_x = grad(q4_eq, x)[0]
	q4_eq_y = grad(q4_eq, y)[0]

	q5_eq_t = grad(q5_eq, t)[0]
	# q5_eq_x = grad(q5_eq, x)[0]
	# q5_eq_y = grad(q5_eq, y)[0]

	q6_eq_t = grad(q6_eq, t)[0]
	# q6_eq_x = grad(q6_eq, x)[0]
	# q6_eq_y = grad(q6_eq, y)[0]

	f4_neq = -tau * (q4_eq_t + c * (q3_eq_x + q2_eq_y))
	f5_neq = -tau * (q5_eq_t + c * (SQRT2*q2_eq_x))
	f6_neq = -tau * (q6_eq_t + c * (SQRT2*q3_eq_y))

	u_x = grad(u,x)[0]
	v_x = grad(v,x)[0]
	rho_x = grad(rho,x)[0]

	f4_neq = SCALE * f4_neq
	f5_neq = SCALE * f5_neq
	f6_neq = SCALE * f6_neq

	return rho_x, u_x, v_x, n1-f4_neq, n2-f5_neq, n3-f6_neq


def data(ruv):
	rho_d, u_d, v_d = ruv[:,0:1], ruv[:,1:2], ruv[:,2:3]
	rho_d = torch.from_numpy(rho_d).to('cuda')
	u_d   = torch.from_numpy(u_d).to('cuda')
	v_d   = torch.from_numpy(v_d).to('cuda')
	def data_res(U, t, x, y):
		rho, u, v, n1, n2, n3 = U
				# q1[q1<0.1] = 0.1
		rho = q1
		u = q2 * c / rho
		v = q3 * c / rho
		return rho_d-rho, u_d-u, v_d-v
	return data_res


def initial(U, t, x, y):
	rho, u, v, n1, n2, n3 = U

	return rho-rBar, u-uBar, v-vBar


# Define the pde domain
pde_domain = domain.Time(tstart, tfinal) * domain.HyperCube([(xmin, xmax), (ymin, ymax)], n_coll=7500)
pde_col    = pde_domain
pde_points = pde_col.generate().get()

wall_1_col = domain.Time(tstart, tfinal) * domain.Line((xmin,ymin), (xmax,ymin), n_coll=1000)
wall_2_col = domain.Time(tstart, tfinal) * domain.Line((xmax,ymin), (xmax,ymax), n_coll=1000)
wall_3_col = domain.Time(tstart, tfinal) * domain.Line((xmax,ymax), (xmin,ymax), n_coll=1000)
wall_4_col = domain.Time(tstart, tfinal) * domain.Line((xmin,ymax), (xmin,ymin), n_coll=1000)

# Define IC Domain
ic_domain   = domain.Time(tstart, tstart) * domain.HyperCube([(xmin, xmax), (ymin, ymax)], n_coll=5000)
ic_col      = ic_domain
ic_points   = ic_col.generate().get()
ic_res      = Residual(ic_points, initial)

wall_1_points = wall_1_col.generate().get()
wall_2_points = wall_2_col.generate().get()
wall_3_points = wall_3_col.generate().get()
wall_4_points = wall_4_col.generate().get()


# # # plt.scatter(wall_1_points[:,0:1], wall_1_points[:,1:2])
# # # plt.scatter(wall_2_points[:,0:1], wall_2_points[:,1:2])
# # # plt.scatter(wall_3_points[:,0:1], wall_3_points[:,1:2])
# # # plt.scatter(wall_4_points[:,0:1], wall_4_points[:,1:2])
# # # plt.scatter(pde_points[:,0:1], pde_points[:,1:2])

# fig, ax = plt.subplots(1,1,subplot_kw={'projection': '3d'})

# ax.scatter(wall_1_points[:,0:1], wall_1_points[:,1:2], wall_1_points[:,2:3])
# ax.scatter(wall_2_points[:,0:1], wall_2_points[:,1:2], wall_2_points[:,2:3])
# ax.scatter(wall_3_points[:,0:1], wall_3_points[:,1:2], wall_3_points[:,2:3])
# ax.scatter(wall_4_points[:,0:1], wall_4_points[:,1:2], wall_4_points[:,2:3])
# # fig.set_size_inches(11, 4)
# ax.scatter(pde_points[:,0:1], pde_points[:,1:2], pde_points[:,2:3])
# # ax.scatter(ic_points[:,0:1], ic_points[:,1:2], ic_points[:,2:3])

# plt.show()
# sys.exit(1)

pde_res    = Residual(pde_points, pde_residual)
wall_1_res = Residual(wall_1_points, wall)
wall_2_res = Residual(wall_2_points, outflow)
wall_3_res = Residual(wall_3_points, wall)
wall_4_res = Residual(wall_4_points, inflow)

wall_1_pde = Residual(wall_1_points, pde_residual)
wall_2_pde = Residual(wall_2_points, pde_residual)
wall_3_pde = Residual(wall_3_points, pde_residual)
wall_4_pde = Residual(wall_4_points, pde_residual)


if __name__ == '__main__':

	model = BNS(tau, c)

	if not steady:
		loader = make_loader(
			[pde_res, wall_1_res, wall_2_res, wall_3_res, wall_4_res, wall_1_pde, wall_2_pde, wall_3_pde, wall_4_pde, ic_res],
			fully_loaded=True, batched=False)
	else:
		loader = make_loader(
			[pde_res, wall_1_res, wall_2_res, wall_3_res, wall_4_res],
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