from remin.solver import Solver, make_trainer
from remin.residual import Residual, make_loader
from remin.solver.residual_loss import EagerLoss
from remin import callbacks
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
	# file_name = 'mesh_w25_lr5e5/mesh_square_best.pt'
	file_name = 'mesh_w25_lr5e5/mesh_square_final.pt'

# Soft model instance
model = Mesh()
mdata = torch.load(file_name)
model.load_state_dict(mdata['model_state_dict'])

meshdir = 'square.msh'
U_p, distance = impose_hard_bc(model, meshdir)

def pde_residual_X(grads, x, y):
    _, _, _, Y_yx, X_xx, X_yy, _, _= grads
    return lambda_a*(X_xx + Y_yx) + mu_a*(2*X_xx + X_yy + Y_yx)

def pde_residual_Y(grads, x, y):
    _, _, X_xy, _, _, _, Y_xx, Y_yy = grads
    return mu_a*(X_xy + Y_xx + 2*Y_yy) + lambda_a*(X_xy + Y_yy)

pde_res = Residual(U_p, [pde_residual_X, pde_residual_Y])

if __name__ == '__main__':

	distance = torch.from_numpy(distance).to(device)
	model = MeshHardBC(distance)
	loader = make_loader(
		[pde_res],
		fully_loaded=True)

	epochs = 5000
	lr = 1e-5

	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	resloss = EagerLoss(nn.MSELoss())
	metloss = EagerLoss(nn.HuberLoss())
	trainer = make_trainer(loader,
	                       optimizer=optimizer,
	                       residual_loss=resloss,
	                       metric_loss=metloss)

	solver = Solver(model,
	                name='mesh_square',
	                save_folder='./mesh_hard',
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
