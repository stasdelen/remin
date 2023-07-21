from remin.solver import Solver, make_trainer
from remin.residual import Residual, make_loader
from remin.solver.residual_loss import EagerLoss
from remin import callbacks
from remin.func import grad
import torch
import numpy as np
from torch import nn
from hard_model import MeshHardBC
from soft_model import Mesh
import argparse
from pre_process import impose_hard_bc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
torch.set_float32_matmul_precision('high')
torch.manual_seed(0)
np.random.seed(0)

# Lame Parameters
mu_a = 0.35
lambda_a = 1.0

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--soft', type=str)

args = parser.parse_args()
file_name = args.soft
if args.soft is None:
	# file_name = 'outputs/mesh_w25_lr5e5/mesh_soft_best.pt'
	file_name = 'outputs/mesh_w25_lr5e5/mesh_soft_final.pt'

# Soft model instance
model = Mesh()
mdata = torch.load(file_name)
model.load_state_dict(mdata['model_state_dict'])

meshdir = 'square.msh'
U_p, distance = impose_hard_bc(model, meshdir)

def pde_residual(U, x, y):
    X, Y = U
    X_x, X_y = grad(X, [x, y])
    Y_x, Y_y = grad(Y, [x, y])
    X_xx, X_xy = grad(X_x, [x, y])
    Y_yx, Y_yy = grad(Y_y, [x, y])
    X_yy = grad(X_y, y)[0]
    Y_xx = grad(Y_x, x)[0]
    
    fx = lambda_a*(X_xx + Y_yx) + mu_a*(2*X_xx + X_yy + Y_yx)
    fy = mu_a*(X_xy + Y_xx + 2*Y_yy) + lambda_a*(X_xy + Y_yy)
    return fx, fy

pde_res = Residual(U_p, pde_residual)

if __name__ == '__main__':

	distance = torch.from_numpy(distance).to(device)
	model = MeshHardBC(distance)
	loader = make_loader(
		[pde_res],
		fully_loaded=True)

	epochs = 10000
	lr = 1e-5

	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	resloss = EagerLoss(nn.MSELoss())
	metloss = EagerLoss(nn.HuberLoss())
	trainer = make_trainer(loader,
	                       optimizer=optimizer,
	                       residual_loss=resloss,
	                       metric_loss=metloss)

	solver = Solver(model,
	                name='mesh_hard',
	                save_folder='./outputs/mesh_hard',
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
