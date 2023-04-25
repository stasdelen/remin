import torch
from model import Heat
import matplotlib.pyplot as plt
from matplotlib import cm


model = Heat()
mdata = torch.load('outputs/heat_best_model.pt')
model.load_state_dict(mdata['model_state_dict'])
model.eval()

t = torch.linspace(0, 1, 100)
x = torch.linspace(0, 1, 100)
T, X = torch.meshgrid(t, x, indexing='xy')

u_pred = model(
    T.flatten()[:, None],
    X.flatten()[:, None]
    ).reshape_as(T)

fig, ax1 = plt.subplots(1,1,subplot_kw={'projection': '3d'})
fig.set_size_inches(6, 4)
fig.suptitle('Heat Equation Solution')

ax1.set(xlabel='x', ylabel='t', zlabel='u')

surf = ax1.plot_surface(
    X.detach(),
    T.detach(),
    u_pred.detach(),
    linewidth=0, antialiased=False, cmap=cm.coolwarm)

fig.savefig('outputs/heat.png', dpi=300)