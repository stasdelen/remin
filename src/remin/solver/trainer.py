import torch
from remin.residual import Loader
from remin.solver.residual_loss import ResidualLoss
from ..residual import Loader, FullLoader, BatchLoader
from .residual_loss import FuncLoss


class Trainer:

    def __init__(self,
                 loader: Loader,
                 model: torch.nn.Module = None,
                 optimizer: torch.optim.Optimizer = None,
                 residual_loss: ResidualLoss = None,
                 metric_loss: ResidualLoss = None) -> None:
        self.model = model
        self.device = None
        self.params = None

        self.loader = loader
        self.optimizer = optimizer
        self.residual_loss = residual_loss
        self.metric_loss = metric_loss

    def setup(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.params = dict(model.named_parameters())
        self.residual_loss.setup(self.model, self.loader)
        if self.metric_loss:
            self.metric_loss.setup(self.model, self.loader)

    def __call__(self) -> float:
        raise NotImplementedError('Train Loop must be defined.')

    def eval_loss(self) -> float:
        raise NotImplementedError('Loss evaluation must be defined.')

    def optim_closure(self, xs):

        def closure():
            self.optimizer.zero_grad()
            resloss = self.residual_loss(self.params, xs)
            resloss.backward()
            return resloss

        return closure


class FullyLoadedTrainer(Trainer):

    def setup(self, model):
        super().setup(model)
        self.loader.instanciate(self.device)

    def __call__(self) -> float:
        resloss = self.residual_loss(self.params, self.loader.xs).item()
        self.optimizer.step(self.optim_closure(self.loader.xs))
        return resloss

    def eval_loss(self) -> float:
        self.model.eval()
        metloss = self.metric_loss(self.params, self.loader.xs).item()
        self.model.train()
        return metloss


class BatchedTrainer(Trainer):

    def __call__(self) -> float:
        resloss = 0.0
        for batch in self.loader.loader:
            xs = torch.vsplit(batch.to(self.device, non_blocking=True),
                              self.loader.vsplit)
            resloss += self.residual_loss(self.params, xs).item()
            self.optimizer.step(self.optim_closure(xs))
        resloss /= self.loader.batch_sampler.n_batches
        return resloss

    def eval_loss(self) -> float:
        metloss = 0.0
        self.model.eval()
        for batch in self.loader.loader:
            xs = torch.vsplit(batch.to(self.device, non_blocking=True),
                              self.loader.vsplit)
            metloss += self.metric_loss(self.params, xs).item()
        metloss /= self.loader.batch_sampler.n_batches
        self.model.train()
        return metloss


def make_trainer(loader: Loader,
                 model: torch.nn.Module = None,
                 optimizer: torch.optim.Optimizer = None,
                 lossfunc=None,
                 residual_loss=None,
                 metric_loss=None):
    if residual_loss is None:
        residual_loss = FuncLoss(lossfunc, model, loader)
    if isinstance(loader, BatchLoader):
        return BatchedTrainer(loader, model, optimizer, residual_loss, metric_loss)
    if isinstance(loader, FullLoader):
        return FullyLoadedTrainer(loader, model, optimizer, residual_loss, metric_loss)
    raise ValueError('Unknown Loader Type.')
