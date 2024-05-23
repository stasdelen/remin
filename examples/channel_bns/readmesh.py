import numpy as np
import sys

def readMesh(fileDir):
	with open(fileDir,'r') as f:
		# read number of nodes in mesh
		while True:
			line = f.readline()
			if '$Nodes' in line:
				break
		Nnodes = (int(f.readline()))

		# Initialize the vertex arrays
		Vx = np.empty((Nnodes, 1), dtype=np.float32)
		Vy = np.empty((Nnodes, 1), dtype=np.float32)

		# Fill the vertex arrays
		for _ in range(Nnodes):
			line = f.readline().split(' ')
			V_id = int(line[0]) - 1
			Vx[V_id, 0] = float(line[1])
			Vy[V_id, 0] = float(line[2])

		# Read Number of elements
		while True:
			line = f.readline()
			if '$Elements' in line:
				break
		Nelements = (int(f.readline()))

		# Find number of triangles
		NboundaryFaces = 0
		while int(f.readline().split(' ')[1]) != 2:
			pos = f.tell()
			NboundaryFaces += 1
		Ntriangles = Nelements - NboundaryFaces

		EtoV = np.empty((Ntriangles, 3), dtype=int)
		Ex = np.empty((Ntriangles, 3), dtype=np.float32)
		Ey = np.empty((Ntriangles, 3), dtype=np.float32)

		# Write EtoV, Ex and Ey
		f.seek(pos)
		for i in range(Ntriangles):
			line = f.readline().split(' ')
			EtoV[i, 0] = int(line[5]) - 1
			EtoV[i, 1] = int(line[6]) - 1
			EtoV[i, 2] = int(line[7]) - 1

			Ex[i, 0] = Vx[int(line[5])-1, 0]
			Ex[i, 1] = Vx[int(line[6])-1, 0]
			Ex[i, 2] = Vx[int(line[7])-1, 0]

			Ey[i, 0] = Vy[int(line[5])-1, 0]
			Ey[i, 1] = Vy[int(line[6])-1, 0]
			Ey[i, 2] = Vy[int(line[7])-1, 0]

		return Vx, Vy, Ex, Ey, EtoV, Ntriangles, Nnodes