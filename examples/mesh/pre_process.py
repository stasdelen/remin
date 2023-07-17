import torch
import numpy as np
from readmesh import readMesh, createPoints

def impose_hard_bc(soft_model, meshdir):

	model = soft_model
	model.eval()

	_, _, _, _, res_col, WALL_1_id, WALL_2_id, WALL_3_id, WALL_4_id = createPoints(meshdir)
	Vx, Vy, _, _, _, _, Nnodes = readMesh(meshdir)

	X_c = Vx
	Y_c = Vy

	# Convert to torch
	res_col = torch.from_numpy(res_col)

	XY = torch.from_numpy(np.hstack((X_c, Y_c))).to('cuda')
	U_soft = model.calc(XY) # will be the input of the second network
	# Create particular solution
	X_p = U_soft[:,0:1].cpu().detach().numpy()
	Y_p = U_soft[:,1:2].cpu().detach().numpy()

	# Stop the loop if it finishes loopng the boundaries -AA
	for i in range(Nnodes):
		if i in WALL_4_id:
			X_p[i, :] = X_c[i, :]
			Y_p[i, :] = Y_c[i, :] - 0.25*np.sin(np.pi * X_c[i, :])
		elif any(i in ls for ls in [WALL_1_id, WALL_2_id, WALL_3_id]):
			X_p[i, :] = X_c[i, :]
			Y_p[i, :] = Y_c[i, :]


	U_p = np.hstack((X_p, Y_p))

	distance = np.zeros((Nnodes,1))

	lb = [min(Vx), min(Vy)]
	ub = [max(Vx), max(Vy)]

	for i in range(Nnodes):
		x_dist = min(abs((Vx[i, :] - lb[0])), abs((Vx[i, :] - ub[0])))
		y_dist = min(abs((Vy[i, :] - lb[1])), abs((Vy[i, :] - ub[1])))
		distance[i,:]= min(x_dist, y_dist)

	# U_p and distance are numpy arrays
	return U_p, distance