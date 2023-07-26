import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

def main(model, name):
    F = 300
    t = torch.linspace(0, 1, F)
    x = torch.linspace(0, 1, 100)
    y = torch.linspace(0, 1, 100)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)

    fig, ax = plt.subplots()
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', '5%', '5%')
    ax.set(xlabel = 'x', ylabel = 'y')
    fig.suptitle('Temperature Distribution')

    T = torch.zeros_like(X)
    frame0 = model(T, X, Y)
    sc = ax.scatter(X.cpu().detach(),
               Y.cpu().detach(),
               c = frame0.cpu().detach(),
               cmap='rainbow',
               animated = True
    )
    cb = fig.colorbar(sc, cax=cax)

    def animate(i):
        T = torch.ones_like(X) * t[i]
        frame = model(T, X, Y)
        sc.set_array(frame.cpu().detach()[:,0])
        return sc,

    ani = animation.FuncAnimation(fig, animate, frames=F,
                                  interval=50,
                                  #repeat_delay=1000,
                                  blit = True)

    writer = animation.FFMpegWriter(
        fps=60, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(name + '.mp4', writer=writer)

    #writer = animation.PillowWriter(fps=60)
    #ani.save(name + '.gif', writer=writer)