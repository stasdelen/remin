import torch
from model import Shrodinger
import matplotlib.pyplot as plt
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str)

    args = parser.parse_args()
    file_name = args.file
    if args.file is None:
        file_name = 'outputs/schrodinger_best.pt'

    x = torch.linspace(-10, 10, 1000).reshape((-1, 1))

    model = Shrodinger()
    mdata = torch.load(file_name)
    model.load_state_dict(mdata['model_state_dict'])
    model.eval()

    phi = model(x)
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(11, 5)
    fig.suptitle('Schrödinger\'s Equation Solution')
    ax.plot(x.detach(), phi.detach(), 'b-', label='Prediction')
    ax.legend(loc='best')
    ax.grid(True)

    fig.savefig(file_name.split('.')[0] + '.png', dpi=300)