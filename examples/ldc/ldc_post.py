import torch
import numpy as np
from ldc_model import Ldc
import matplotlib.pyplot as plt
import argparse


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--file', type=str)

	args = parser.parse_args()
	file_name = args.file
	if args.file is None:
		file_name = 'outputs/LDC_Re100_best.pt'
		# file_name = 'outputs/LDC_Re100_final.pt'

	model = Ldc()
	mdata = torch.load(file_name)
	model.load_state_dict(mdata['model_state_dict'])
	model.eval()

	x = torch.linspace(0,1,1000)
	y = torch.linspace(0,1,1000)

	X, Y = torch.meshgrid(x, y, indexing='xy')
	X = X.flatten()[:,None].requires_grad_()
	Y = Y.flatten()[:,None].requires_grad_()

	uvp = model.calc(X, Y).cpu().detach().numpy()
	u = uvp[:,0:1] 
	v = uvp[:,1:2]
	p = uvp[:,2:3]
	m = np.sqrt(u*u + v*v)

	
	plt.scatter(X.detach(), Y.detach(), c=m, cmap='jet')
	
	plt.colorbar()
	plt.show()
	plt.savefig(file_name.split('.')[0] + '.png', dpi=300)
