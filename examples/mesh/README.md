# Physics-Informed Neural Networks for Mesh Deformation with Exact Boundary Enforcement

Code accompanying the manuscript titled "Physics-Informed Neural Networks for Mesh Deformation with Exact Boundary Enforcement", authored by Atakan Aygun, Romit Maulik and Ali Karakus.

# Abstract

In this work, we have applied physics-informed neural networks (PINN) for solving mesh deformation problems. We used the collocation PINN method to capture the new positions of the vertex nodes while preserving the connectivity information. We use linear elasticity equations for mesh deformation. To prevent vertex collisions or edge overlap, the mesh movement in this work is conducted in steps with relatively small movements. For moving boundary problems, the exact position of the boundary is essential for having an accurate solution. However, PINNs are frequently unable to satisfy Dirichlet boundary conditions exactly. To overcome this issue, we have used hard boundary condition enforcement to automatically satisfy Dirichlet boundary conditions. Specifically, we first trained a PINN with soft boundary conditions to obtain a particular solution. Then, this solution was tuned with exact boundary positions and a proper distance function by using a new PINN considering only the equation residual. To assess the accuracy of our approach, we used the classical translation and rotation tests and compared them with a proper mesh quality metric considering the change in the element area and shape. The results show the accuracy of this approach is comparable with that of finite element solutions. We also solved different moving boundary problems, resembling commonly used fluid–structure interaction problems. This work provides insight into using PINN for mesh-deformation problems without needing a discretization scheme with reasonable accuracy.

# Citation

	@article{aygun_physics-informed_2023,
	author = {Aygun, Atakan and Maulik, Romit and Karakus, Ali},
	title = {Physics-informed neural networks for mesh deformation with exact boundary enforcement},
	journal = {Engineering Applications of Artificial Intelligence},
	volume = {125},
	pages = {106660},
	year = {2023},
	doi = {10.1016/j.engappai.2023.106660}
	}
