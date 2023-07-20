import torch
import numpy as np
from hard_model import MeshHardBC
from soft_model import Mesh
import matplotlib.pyplot as plt
import argparse
from pre_process import impose_hard_bc
from readmesh import readMesh

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
torch.set_float32_matmul_precision('high')
torch.manual_seed(0)
np.random.seed(0)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-sf', '--soft', type=str)
	parser.add_argument('-hf', '--hard', type=str)

	args = parser.parse_args()
	hard_file = args.hard
	soft_file = args.soft
	if args.hard is None:
		# file_name = 'outputs/mesh_w25_lr5e5/mesh_hard_best.pt'
		hard_file = 'outputs/mesh_hard/mesh_hard_final.pt'
	if args.soft is None:
		# file_name = 'outputs/mesh_w25_lr5e5/mesh_soft_best.pt'
		soft_file = 'outputs/mesh_w25_lr5e5/mesh_soft_final.pt'

	# Load soft model
	soft_model = Mesh()
	mdata = torch.load(soft_file)
	soft_model.load_state_dict(mdata['model_state_dict'])
	soft_model.eval()
	
	# Compute distance
	meshdir = 'square.msh'
	U_p, distance = impose_hard_bc(soft_model, meshdir)
	distance = torch.from_numpy(distance).to(device)

	# Load hard model
	hard_model = MeshHardBC(distance)
	mdata = torch.load(hard_file)
	hard_model.load_state_dict(mdata['model_state_dict'])
	hard_model.eval()

	Vx, Vy, Ex, Ey, EtoV, Ntriangles, Nnodes = readMesh(meshdir)

	XY = torch.from_numpy(np.hstack((Vx, Vy))).to(device)
	U_soft = soft_model.calc(XY)
	X_p = U_soft[:,0:1].cpu().detach().numpy()
	Y_p = U_soft[:,1:2].cpu().detach().numpy()

	# Hard residual guess
	U_p = torch.from_numpy(U_p).to(device)
	U_hard = hard_model.calc(U_p).cpu().detach().numpy()
	

	fig, ax = plt.subplots(1, 1)
	fig.set_size_inches(8, 8)

	ax.scatter(X_p, Y_p, c='b', s = 6)
	ax.scatter(U_hard[:,0], U_hard[:,1], c='r', s = 2)
	

	x_test = np.linspace(0,1)
	ax.plot(x_test, 1 - 0.25 * np.sin(np.pi * x_test))

	plt.show()

	fig.savefig(hard_file.split('.')[0] + '.png', dpi=300)
	