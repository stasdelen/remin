import torch
from ..residual import Residual, Loader


class ResidualLoss:

    def __init__(self,
                 lossfunc,
                 model: torch.nn.Module = None,
                 loader: Loader = None) -> None:
        self.loader = loader
        self.lossfunc = lossfunc
        self.model = model
        self.__name__ = 'residual_loss'

    def setup(self, model, loader):
        self.model = model
        self.loader = loader

    def __call__(self, params, xs) -> float:
        raise NotImplementedError('Loss must be defined.')


class FuncLoss(ResidualLoss):

    def __call__(self, params, xs) -> float:
        resloss = 0
        for i in range(self.loader.n_res):
            domain = Residual.split_domain(xs[i])
            for func in self.loader.functions[i]:
                resloss += self.loader.weights[i] * self.lossfunc(
                    func(params, *domain), torch.zeros_like(domain[0]))
        return resloss


class EagerLoss(ResidualLoss):

    def __call__(self, params, xs) -> float:
        resloss = 0
        for i in range(self.loader.n_res):
            domain = Residual.split_domain(xs[i])
            U = self.model(*domain)
            for func in self.loader.functions[i]:
                resloss += self.loader.weights[i] * self.lossfunc(
                    func(U, *domain), torch.zeros_like(domain[0]))
        return resloss


class ModLoss(ResidualLoss):

    def __call__(self, params, xs) -> float:
        resloss = 0
        for i in range(self.loader.n_res):
            domain = Residual.split_domain(xs[i])
            for func in self.loader.functions[i]:
                resloss += self.loader.weights[i] * self.lossfunc(
                    func(self.model, *domain), torch.zeros_like(domain[0]))
        return resloss
