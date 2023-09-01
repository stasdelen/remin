from time import time
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
import csv


class Callback:
    # Callback interface
    def __init__(self) -> None:
        self.state_dict: Dict[str, Any] = {}

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass


class UnionCallback(Callback):

    def __init__(self, state_dict: Dict[str, Any], *callbacks: List[Callback]) -> None:
        super().__init__()
        self.callbacks = callbacks
        self.state_dict = state_dict
        for callback in callbacks:
            callback.state_dict = state_dict

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()

    def on_epoch_begin(self):
        for callback in self.callbacks:
            callback.on_epoch_begin()

    def on_epoch_end(self):
        rval = 0
        for callback in self.callbacks:
            if callback.on_epoch_end():
                rval = -1
        return rval

    def reset(self, *callbacks):
        self.callbacks = callbacks
        for callback in callbacks:
            callback.state_dict = self.state_dict

    def append(self, *callbacks):
        self.callbacks += callbacks


class TotalTimeCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.t0 = 0

    def on_train_begin(self):
        self.t0 = time()

    def on_train_end(self):
        print(f'\nTotal training time: {((time() - self.t0) / 60):7.3f}mins.')


class LogCallback(Callback):

    def __init__(self, log_epoch=100, log_progress=10) -> None:
        super().__init__()
        self.log_epoch = log_epoch
        self.log_progress = log_progress
        self.t0 = 0
        self.t_ave = 0
        self.metric = False

    def on_train_begin(self):
        self.metric = bool(self.state_dict['solver'].trainer.metric_loss)

    def on_epoch_begin(self):
        self.t0 = time()

    def on_epoch_end(self):
        epoch = self.state_dict['epoch']
        epochs = self.state_dict['epochs']
        resloss = self.state_dict['residual']
        metloss = self.state_dict['metric']

        self.t_ave = (self.t_ave * epoch + (time() - self.t0)) / (epoch + 1)
        if epoch % self.log_progress == 0:
            if resloss < 1e-3:
                prefix = f'Loss: {resloss:.6E}'
            else:
                prefix = f'Loss: {resloss:10.6f}'
            if self.metric:
                if metloss < 1e-3:
                    prefix += f' - Metric Loss: {metloss:.6E}'
                else:
                    prefix += f' - Metric Loss: {metloss:10.6f}'
            self._printProgress(epoch, epochs, self.log_epoch, prefix,
                                f'{(self.t_ave*1e3):7.3f}ms/epoch')

    def on_train_end(self):
        if self.state_dict['epoch'] == self.state_dict['epochs']:
            print(
                f'\nTraining over:\n\tBest Loss Achieved: {self.state_dict["best"]:10.6f}'
            )

    @staticmethod
    def _printProgress(epoch,
                       epochs,
                       log_epoch,
                       prefix='',
                       suffix='',
                       length=20,
                       fill='#',
                       printEnd='\r'):
        filledLength = int(length * (epoch % log_epoch) // log_epoch)
        if epoch % log_epoch == 0:
            filledLength = length
            printEnd = '\n'
        loading_bar = fill * filledLength + '.' * (length - filledLength)
        print(f'{prefix} [{loading_bar}] [{epoch:>5d}/{epochs:>5d}] {suffix}',
              flush=True,
              end=printEnd)


class SaveCallback(Callback):

    def __init__(self, best_suffix='_best.pt', final_suffix='_final.pt') -> None:
        super().__init__()
        self.best_loss = float('inf')
        self.best_suffix = best_suffix
        self.final_suffix = final_suffix

    def on_epoch_end(self):
        resloss = self.state_dict['residual']
        epoch = self.state_dict['epoch']
        solver = self.state_dict['solver']
        if resloss < self.best_loss:
            self.best_loss = resloss
            solver.save(epoch, self.best_suffix)

    def on_train_end(self):
        solver = self.state_dict['solver']
        epoch = self.state_dict['epoch']
        solver.save(epoch, self.final_suffix)


class PlotCallback(Callback):

    def __init__(self,
                 state='residual',
                 size_inches=(10, 5),
                 color='red',
                 linestyle='-',
                 linewidth=2,
                 title='Residual Loss(L)',
                 ylabel='Loss',
                 xlabel='Epoch',
                 yscale='log',
                 grid_on=True,
                 name='L_vs_epoch.png',
                 dpi=300) -> None:
        super().__init__()
        self.state = state
        self.size_inches = size_inches
        self.color = color
        self.linestyle = linestyle
        self.linewidth = linewidth
        self.title = title
        self.ylabel = ylabel
        self.xlabel = xlabel
        self.yscale = yscale
        self.fig_name = name
        self.grid_on = grid_on
        self.dpi = dpi
        self.losses = None
        self.save_path = None

    def on_train_begin(self):
        self.losses = np.zeros((self.state_dict['epochs'], ))
        self.save_path = self.state_dict['solver'].save_path

    def on_epoch_end(self):
        self.losses[self.state_dict['epoch'] - 1] = self.state_dict[self.state]

    def on_train_end(self):
        fig, ax = plt.subplots(1, 1)
        if self.size_inches:
            fig.set_size_inches(*self.size_inches)
        ax.plot(self.losses[:self.state_dict['epoch']],
                color=self.color,
                linestyle=self.linestyle,
                linewidth=self.linewidth)
        ax.set_yscale(self.yscale)
        ax.set(title=self.title, ylabel=self.ylabel, xlabel=self.xlabel)
        ax.grid(self.grid_on)
        plt.savefig(self.save_path + self.fig_name, dpi=self.dpi)


class CSVCallback(Callback):

    def __init__(self, name: str = 'training_data.csv') -> None:
        super().__init__()
        self.name = name
        self.fields = ['epoch', 'residual', 'metric']

    def on_train_begin(self):
        self.data = np.zeros((self.state_dict['epochs'], len(self.fields)))
        self.save_path = self.state_dict['solver'].save_path

    def on_epoch_end(self):
        for i, field in enumerate(self.fields):
            self.data[self.state_dict['epoch'] - 1, i] = self.state_dict[field]

    def on_train_end(self):
        with open(self.save_path + '/' + self.name, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)

            csvwriter.writerow(self.fields)
            csvwriter.writerows(self.data[:self.state_dict['epoch']])


class EarlyStoppingCallback(Callback):

    def __init__(self, min_delta=0, patience=8, restore_best_weights=False) -> None:
        super().__init__()
        self.min_delta = min_delta
        self.max_patience = patience
        self.patience = 0
        self.restore_best_weights = restore_best_weights
        self.saved_loss = float('inf')
        self.best_loss = float('inf')

    def on_epoch_end(self):
        resloss = self.state_dict['residual']

        if self.patience >= self.max_patience:
            print(
                f'''\nEarly Stopping at {self.state_dict['epoch']}/{self.state_dict['epochs']}:
        Residual Loss: {resloss:10.6f}
        Saved Loss:    {self.saved_loss:10.6f}
        Best Loss:     {self.best_loss:10.6f}''')
            return 1

        if (self.saved_loss - resloss) / resloss * 100 > self.min_delta:
            self.patience = 0
            if self.restore_best_weights:
                self.saved_loss = self.state_dict['best']
            else:
                self.saved_loss = resloss
        else:
            #print(resloss - self.saved_loss)
            self.patience += 1


class ToleranceCallback(Callback):

    def __init__(self, tolerance) -> None:
        super().__init__()
        self.tolerance = tolerance

    def on_epoch_end(self):
        resloss = self.state_dict['residual']

        if self.tolerance >= resloss:
            print(
                f'''\nStopping at {self.state_dict['epoch']}/{self.state_dict['epochs']}:
        Residual Loss: {resloss:10.6f}
        ''')
            return 1
