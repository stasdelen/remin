import torch
from ..residual import Residual, Loader


class ResidualLoss:

    def __init__(self,
                 lossfunc,
                 model: torch.nn.Module = None,
                 loader: Loader = None) -> None:
        self.loader = loader
        self.lossfunc = lossfunc
        self.lossfunc.reduction = 'none'
        self.model = model
        self.__name__ = 'residual_loss'

    def setup(self, model, loader):
        self.model = model
        self.loader = loader

    def __call__(self, params, xs) -> float:
        raise NotImplementedError('Loss must be defined.')


class FuncLoss(ResidualLoss):

    def __call__(self, params, xs) -> float:
        for i in range(self.loader.n_res):
            domain = Residual.split_domain(xs[i])
            residuals = torch.hstack(self.loader.functions[i](params, *domain))
            resloss = torch.mean(
                self.loader.weights[i] *
                self.lossfunc(residuals, torch.zeros_like(domain[0])), 0).sum()
        return resloss


class EagerLoss(ResidualLoss):

    def __call__(self, params, xs) -> float:
        resloss = 0
        for i in range(self.loader.n_res):
            domain = Residual.split_domain(xs[i])
            U = self.model(*domain)
            residuals = torch.hstack(self.loader.functions[i](U, *domain))
            resloss += torch.mean(
                self.loader.weights[i] *
                self.lossfunc(residuals, torch.zeros_like(residuals)), 0).sum()
        return resloss


class ModLoss(ResidualLoss):

    def __call__(self, params, xs) -> float:
        resloss = 0
        for i in range(self.loader.n_res):
            domain = Residual.split_domain(xs[i])
            residuals = torch.hstack(self.loader.functions[i](self.model, *domain))
            resloss += torch.mean(
                self.loader.weights[i] *
                self.lossfunc(residuals, torch.zeros_like(residuals)), 0).sum()
        return resloss
