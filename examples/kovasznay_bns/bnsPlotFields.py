import numpy as np
import sys

def bns_plot_fields(fileName, Vx, Vy, Ex, Ey, Ntriangles, Nnodes,
										field_u, field_v, field_rho):
	with open(fileName, 'w') as f:
		f.write("<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">\n")
		f.write("  <UnstructuredGrid>\n")
		f.write("    <Piece NumberOfPoints=\"" + str(Ntriangles*3) + "\" NumberOfCells=\"" + str(Ntriangles) + "\">\n")

		# Write out nodes
		f.write("      <Points>\n")
		f.write("        <DataArray type=\"Float32\" NumberOfComponents=\"3\" Format=\"ascii\">\n")

		for i in range(Ntriangles):
			for j in range(3):
				f.write("       ")
				f.write("{} {} {}\n".format(Ex[i,j], Ey[i,j], 0))

		
		f.write("        </DataArray>\n")
		f.write("      </Points>\n")

		f.write("      <PointData Scalars=\"scalars\">\n")

		# Write out velocity
		f.write("        <DataArray type=\"Float32\" Name=\"Velocity\" NumberOfComponents=\"2\" Format=\"ascii\">\n")
		for i in range(Ntriangles):
			for j in range(3):
				f.write("       ")
				f.write("       ")
				f.write("{} {}\n".format(field_u[i, j], field_v[i, j]))
		f.write("        </DataArray>\n")

		# Write out pressure
		f.write("        <DataArray type=\"Float32\" Name=\"Density\" Format=\"ascii\">\n")
		for i in range(Ntriangles):
			for j in range(3):
				f.write("       ")
				f.write("       ")
				f.write("{}\n".format(field_rho[i, j]))
		f.write("        </DataArray>\n")

		f.write("      </PointData>\n")

		f.write("    <Cells>\n")
		f.write("      <DataArray type=\"Int32\" Name=\"connectivity\" Format=\"ascii\">\n")
		cnt = 0
		for _ in range(Ntriangles):
			f.write("       {} {} {}\n".format(cnt, cnt+1, cnt+2))
			cnt = cnt+3

		f.write("        </DataArray>\n")

		f.write("        <DataArray type=\"Int32\" Name=\"offsets\" Format=\"ascii\">\n")
		cnt=3
		for _ in range(Ntriangles):
			f.write("       {}\n".format(cnt))
			cnt = cnt+3
		f.write("       </DataArray>\n")

		f.write("        <DataArray type=\"Int32\" Name=\"types\" Format=\"ascii\">\n")
		for _ in range(Ntriangles):
			f.write("5\n")
		f.write("        </DataArray>\n")

		f.write("      </Cells>\n")
		f.write("    </Piece>\n")
		f.write("  </UnstructuredGrid>\n")
		f.write("</VTKFile>\n")
