import torch
import numpy as np
from channel_model import BNS
from channel_config import *
import matplotlib.pyplot as plt
import argparse
from readmesh import readMesh
from bnsPlotFields import bns_plot_fields


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--file', type=str)

	args = parser.parse_args()
	file_name = args.file
	if args.file is None:
		file_name = save_folder + '/bns_best.pt'
	
	model = BNS(tau, c)
	mdata = torch.load(file_name)
	model.load_state_dict(mdata['model_state_dict'])
	model.eval()

	Vx, Vy, Ex, Ey, EtoV, Ntriangles, Nnodes = readMesh(mesh_file)

	if not steady: 
		F = 51
		t = np.linspace(0,tfinal,F)
		for i in range(F):
			T = t[i] * torch.ones_like(torch.Tensor(Vx))

			uvp = model.calc(T, torch.Tensor(Vx), torch.Tensor(Vy)).cpu().detach().numpy()
			rho_PINN = uvp[:,0:1]
			u_PINN = uvp[:,1:2] 
			v_PINN = uvp[:,2:3]

			field_u = np.empty((Ntriangles, 3))
			field_v = np.empty((Ntriangles, 3))
			field_rho = np.empty((Ntriangles, 3))
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

			out_name = 'BNS_channel_' + str(i) + '.vtu'
			bns_plot_fields(out_name, Vx, Vy, Ex, Ey, Ntriangles, Nnodes,
											field_u, field_v, field_rho)

	else:
		uvp = model.calc(torch.Tensor(Vx), torch.Tensor(Vy)).cpu().detach().numpy()
		rho_PINN = uvp[:,0:1]
		u_PINN = uvp[:,1:2] 
		v_PINN = uvp[:,2:3]

		field_u = np.empty((Ntriangles, 3))
		field_v = np.empty((Ntriangles, 3))
		field_rho = np.empty((Ntriangles, 3))
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

		out_name = 'BNS_channel_steady' + '.vtu'
		bns_plot_fields(out_name, Vx, Vy, Ex, Ey, Ntriangles, Nnodes,
										field_u, field_v, field_rho)

