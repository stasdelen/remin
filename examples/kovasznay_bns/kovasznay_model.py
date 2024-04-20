import torch
from torch import nn

# class adaptive_layer(nn.Module):
#   def __init__(self):
#     super().__init__()

#   def forward(self, x):
#     self.A = nn.Parameter(torch.tensor(0.05), requires_grad=True)
#     return torch.tanh(self.A*x)

class BNS(nn.Module):
  def __init__(self, tau, c):
    super().__init__()
    self.tau = tau
    self.c   = c
    self.network_eq = nn.Sequential(
      nn.Linear(2, 40),
      nn.Tanh(),
      nn.Linear(40, 40),
      nn.Tanh(),
      nn.Linear(40, 40),
      nn.Tanh(),
      nn.Linear(40, 40),
      nn.Tanh(),
      nn.Linear(40, 40),
      nn.Tanh(),
      nn.Linear(40, 3)
      )

    self.network_Neq = nn.Sequential(
              nn.Linear(2, 40),
              nn.Tanh(),
              nn.Linear(40, 40),
              nn.Tanh(),
              nn.Linear(40, 40),
              nn.Tanh(),
              nn.Linear(40, 40),
              nn.Tanh(),
              nn.Linear(40, 40),
              nn.Tanh(),
              nn.Linear(40, 3)
        )

  def forward(self, x, y):
    network_in = torch.hstack((x, y))
    out_eq  = self.network_eq(network_in)
    out_Neq = self.network_Neq(network_in)
    rho, u, v = out_eq[:,0:1], out_eq[:,1:2], out_eq[:,2:3]
    n1, n2, n3 = out_Neq[:,0:1], out_Neq[:,1:2], out_Neq[:,2:3]

    return rho, u, v, n1, n2, n3

  def calc(self, x, y):
    network_in = torch.hstack((x, y))
    out_eq  = self.network_eq(network_in)
    # out_Neq = self.network_Neq(network_in)
    rho, u, v = out_eq[:,0:1], out_eq[:,1:2], out_eq[:,2:3]
    # rho = q1
    # u   = q2 * self.c / q1
    # v   = q3 * self.c / q1
    return torch.hstack((rho, u, v))