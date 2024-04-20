import torch
import numpy as np
from kovasznay_model import BNS
from kovasznay_config import *
import matplotlib.pyplot as plt
import argparse
from readmesh import readMesh
from bnsPlotFields import bns_plot_fields
import sys

lambda_ = Re/2 - np.sqrt(Re*Re/4 + 4*np.pi**2)
PI = np.pi

def exact(x, y):

	u = u0 * (1 - torch.exp(lambda_*x)*torch.cos(2*PI*y))
	v = u0 * (lambda_/(2*PI) * torch.exp(lambda_*x)*torch.sin(2*PI*y))

	return u, v

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


	uvp = model.calc(torch.Tensor(Vx), torch.Tensor(Vy)).cpu().detach().numpy()
	rho_PINN = uvp[:,0:1]
	u_PINN = uvp[:,1:2] 
	v_PINN = uvp[:,2:3]

	u_exact, v_exact = exact(torch.Tensor(Vx), torch.Tensor(Vy))
	r_exact = np.ones_like(u_exact);

	print("L2 Norm of the u velocity error:", np.linalg.norm(u_exact-u_PINN))
	print("L2 Norm of the v velocity error:", np.linalg.norm(v_exact-v_PINN))
	print("L2 Norm of the density error:", np.linalg.norm(r_exact-rho_PINN))

	sys.exit(1)

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

	out_name = 'BNS_Kovasznay' + '.vtu'
	bns_plot_fields(out_name, Vx, Vy, Ex, Ey, Ntriangles, Nnodes,
									field_u, field_v, field_rho)

	out_name = 'BNS_Kovasznay_exact' + '.vtu'
	bns_plot_fields(out_name, Vx, Vy, Ex, Ey, Ntriangles, Nnodes,
									field_u_exact, field_v_exact, field_r_exact)

	out_name = 'BNS_Kovasznay_error' + '.vtu'
	bns_plot_fields(out_name, Vx, Vy, Ex, Ey, Ntriangles, Nnodes,
									field_u_exact-field_u, field_v_exact-field_v, field_r_exact-field_rho)
