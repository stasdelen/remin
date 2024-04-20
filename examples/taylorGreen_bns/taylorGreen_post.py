import torch
import numpy as np
from taylorGreen_model import BNS
from taylorGreen_config import *
import matplotlib.pyplot as plt
import argparse
from readmesh import readMesh
from bnsPlotFields import bns_plot_fields

def exact(t, x, y):
	u = -torch.cos(x)*torch.sin(y)*torch.exp(-2*t*nu)
	v =  torch.sin(x)*torch.cos(y)*torch.exp(-2*t*nu)

	return u, v


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--file', type=str)

	args = parser.parse_args()
	file_name = args.file
	if args.file is None:
		file_name = save_folder + '/bns_best.pt'
	
	model = BNS()
	mdata = torch.load(file_name)
	model.load_state_dict(mdata['model_state_dict'])
	model.eval()

	Vx, Vy, Ex, Ey, EtoV, Ntriangles, Nnodes = readMesh(mesh_file)

	F = 51
	t = np.linspace(0,tfinal,F)
	for i in range(F):
		T = t[i] * torch.ones_like(torch.Tensor(Vx))

		uvp = model.calc(T, torch.Tensor(Vx), torch.Tensor(Vy)).cpu().detach().numpy()
		rho_PINN = uvp[:,0:1]
		u_PINN = uvp[:,1:2] 
		v_PINN = uvp[:,2:3]
		u_exact, v_exact = exact(T, torch.Tensor(Vx), torch.Tensor(Vy))
		r_exact = np.ones_like(u_exact);

		field_u = np.empty((Ntriangles, 3))
		field_v = np.empty((Ntriangles, 3))
		field_rho = np.empty((Ntriangles, 3))

		field_u_exact = np.empty((Ntriangles, 3))
		field_v_exact = np.empty((Ntriangles, 3))
		field_r_exact = np.empty((Ntriangles, 3))

		for j in range(Ntriangles):
		  field_u[j, 0] = u_PINN[EtoV[j, 0]]
		  field_u[j, 1] = u_PINN[EtoV[j, 1]]
		  field_u[j, 2] = u_PINN[EtoV[j, 2]]

		  field_v[j, 0] = v_PINN[EtoV[j, 0]]
		  field_v[j, 1] = v_PINN[EtoV[j, 1]]
		  field_v[j, 2] = v_PINN[EtoV[j, 2]]

		  field_rho[j, 0] = rho_PINN[EtoV[j, 0]]
		  field_rho[j, 1] = rho_PINN[EtoV[j, 1]]
		  field_rho[j, 2] = rho_PINN[EtoV[j, 2]]

		  field_u_exact[j, 0] = u_exact[EtoV[j,0]]
		  field_u_exact[j, 1] = u_exact[EtoV[j,1]]
		  field_u_exact[j, 2] = u_exact[EtoV[j,2]]

		  field_v_exact[j, 0] = v_exact[EtoV[j,0]]
		  field_v_exact[j, 1] = v_exact[EtoV[j,1]]
		  field_v_exact[j, 2] = v_exact[EtoV[j,2]]

		  field_r_exact[j, 0] = r_exact[EtoV[j,0]]
		  field_r_exact[j, 1] = r_exact[EtoV[j,1]]
		  field_r_exact[j, 2] = r_exact[EtoV[j,2]]

		out_name = 'BNS_taylorGreen_' + str(i) + '.vtu'
		bns_plot_fields(out_name, Vx, Vy, Ex, Ey, Ntriangles, Nnodes,
										field_u, field_v, field_rho)

		out_name = 'BNS_taylorGreen_exact_' + str(i) + '.vtu'
		bns_plot_fields(out_name, Vx, Vy, Ex, Ey, Ntriangles, Nnodes,
										field_u_exact, field_v_exact, field_r_exact)

		out_name = 'BNS_taylorGreen_error_' + str(i) + '.vtu'
		bns_plot_fields(out_name, Vx, Vy, Ex, Ey, Ntriangles, Nnodes,
										field_u_exact-field_u, field_v_exact-field_v, field_r_exact-field_rho)
