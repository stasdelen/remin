import os
from typing import List
import torch
from ..func import traceable
from ..callbacks import Callback, UnionCallback, SaveCallback, LogCallback
from .trainer import Trainer


class Solver:

    def __init__(self,
                 model: torch.nn.Module,
                 name: str = 'pinn',
                 save_folder: str = '.',
                 trainer: Trainer = None,
                 callbacks: List[Callback] = None) -> None:
        self.name = name
        self.save_path = save_folder + '/'

        self.trainer = trainer
        self.model = model
        self.trainer.setup(self.model)

        self.epoch = 1
        if callbacks is None:
            callbacks = [SaveCallback(), LogCallback()]

        self.state_dict = {
            'solver': self,
            'epoch': None,
            'epochs': None,
            'resloss': None,
            'metloss': None
        }
        self.callback = UnionCallback(self.state_dict, *callbacks)

    def load(self, filename):
        assert self.model is not None
        state_dict = torch.load(filename)
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.trainer.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.trainer.residual_loss.lossfunc = state_dict['loss']

    def save(self, epoch, suffix=''):
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
        torch.save(
            {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.trainer.optimizer.state_dict(),
                'loss': self.trainer.residual_loss.lossfunc,
            }, self.save_path + self.name + suffix)

    def compile(self, backend='inductor', fullgraph=False):
        if self.trainer.residual_loss is None:
            raise ValueError('Residual Loss must be defined.')
        if self.trainer.residual_loss.lossfunc is None:
            raise ValueError('Loss function must be defined.')

        self.trainer.residual_loss = torch.compile(traceable(
            self.trainer.residual_loss),
                                                   backend=backend,
                                                   fullgraph=fullgraph)

    def update_state(self, **state):
        for key in state.keys():
            self.state_dict[key] = state[key]

    def append_callbacks(self, *callbacks):
        self.callback.append(*callbacks)

    def reset_callbacks(self, *callbacks):
        self.callback.reset(*callbacks)

    def fit(self, epochs=1):
        if self.trainer.residual_loss is None:
            raise ValueError('Residual Loss must be defined.')
        if self.trainer.residual_loss.lossfunc is None:
            raise ValueError('Loss function must be defined.')
        if self.trainer.optimizer is None:
            raise ValueError('Optimizer must be defined.')

        self.update_state(epochs=epochs)

        self.callback.on_train_begin()
        for epoch in range(1, epochs + 1):
            self.callback.on_epoch_begin()
            resloss = self.trainer()
            if self.trainer.metric_loss:
                metloss = self.trainer.eval_loss()
                self.update_state(metloss=metloss)
            self.update_state(resloss=resloss, epoch=epoch)
            if self.callback.on_epoch_end():
                break
        self.callback.on_train_end()
        self.epoch = epoch
