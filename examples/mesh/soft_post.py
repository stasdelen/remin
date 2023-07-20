import torch
import numpy as np
from soft_model import Mesh
import matplotlib.pyplot as plt
import argparse
from readmesh import readMesh


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--file', type=str)

	args = parser.parse_args()
	file_name = args.file
	if args.file is None:
		# file_name = 'outputs/mesh_w25_lr5e5/mesh_soft_best.pt'
		file_name = 'outputs/mesh_w25_lr5e5/mesh_soft_final.pt'
	
	model = Mesh()
	mdata = torch.load(file_name)
	model.load_state_dict(mdata['model_state_dict'])
	model.eval()

	meshdir = 'square.msh'

	Vx, Vy = readMesh(meshdir)[0:2]

	XY = torch.from_numpy(np.hstack((Vx, Vy)))
	U_soft = model.calc(XY)
	X_p = U_soft[:,0:1].cpu().detach().numpy()
	Y_p = U_soft[:,1:2].cpu().detach().numpy()


	fig, ax = plt.subplots(1, 1)
	fig.set_size_inches(8,8)

	ax.scatter(X_p, Y_p)
	x_test = np.linspace(0,1)
	ax.plot(x_test, 1 - 0.25 * np.sin(np.pi * x_test))

	plt.show()

	fig.savefig(file_name.split('.')[0] + '.png', dpi=300)
	

	

