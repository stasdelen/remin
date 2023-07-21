import torch
from torch import nn

class Ldc(nn.Module):
	def __init__(self):
		super().__init__()
		self.network = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 3)
			)

	def forward(self, x, y):
		network_in = torch.hstack((x,y))
		uvp = self.network(network_in)
		return uvp[:,0:1], uvp[:,1:2], uvp[:,2:3]

	def calc(self, x,y):
		uvp = self.network(torch.hstack((x,y)))
		u, v, p = uvp[:,0:1], uvp[:,1:2], uvp[:,2:3]
		return torch.hstack((u, v, p))