import torch
from model import Wave
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str)

    args = parser.parse_args()
    file_name = args.file
    if args.file is None:
        file_name = 'outputs/wave_best.pt'
    
    model = Wave()
    mdata = torch.load(file_name)
    model.load_state_dict(mdata['model_state_dict'])
    model.eval()

    t_size, x_size = 100, 100

    t = torch.linspace(0, 1, t_size)
    x = torch.linspace(0, 1, x_size)
    T, X = torch.meshgrid(t, x, indexing='xy')

    z_predict = model(T.reshape(-1, 1), X.reshape(-1, 1)).reshape(t_size, x_size)

    z_true = torch.cos(3*torch.pi*T) * torch.sin(3*torch.pi*X)

    fig, (ax0, ax1, ax2) = plt.subplots(1,3,subplot_kw={'projection': '3d'})
    fig.set_size_inches(11, 4)
    fig.suptitle('Wave Equation Solution')

    ax0.set(xlabel='x', ylabel='t', zlabel='u', title='Absolute Error')
    ax1.set(xlabel='x', ylabel='t', zlabel='u', title='Prediction')
    ax2.set(xlabel='x', ylabel='t', zlabel='u', title='True Value')

    X, T, z_predict, z_true = X.detach(), T.detach(), z_predict.detach(), z_true.detach()

    surf = ax0.plot_surface(X, T, torch.abs(z_true-z_predict).detach(), linewidth=0, antialiased=False, cmap=cm.coolwarm)
    surf = ax1.plot_surface(X, T, z_predict, linewidth=0, antialiased=False, cmap=cm.coolwarm)
    surf = ax2.plot_surface(X, T, z_true, linewidth=0, antialiased=False, cmap=cm.coolwarm)

    fig.savefig(file_name.split('.')[0] + '.png', dpi=300)