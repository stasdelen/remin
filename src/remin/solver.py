import torch
from .func import traceable
from .residual import ResidualLoader, Residual
from time import thread_time


class Solver:

    def __init__(self, name, model, residual_loader: ResidualLoader, optimizer,
                 lossfunc) -> None:
        self.name = name
        self.model = model
        self.device = next(model.parameters()).device
        self.params = dict(model.named_parameters())

        self.residual_loader = residual_loader
        self.residual_loader.instanciate(self.device)
        if self.residual_loader.fully_loaded:
            self.train_loop = self._train_loop_fullloaded
        else:
            self.train_loop = self._train_loop_dataloader

        self.optimizer = optimizer
        self.lossfunc = lossfunc
        self.residual_loss = None

    def load(self, filename):
        assert self.model is not None
        state_dict = torch.load(filename)
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.lossfunc = state_dict['loss']

    def save(self, epoch, name):
        torch.save(
            {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.lossfunc,
            }, name)

    def compile(self, backend='inductor', fullgraph=False):
        if self.lossfunc is None:
            raise ValueError('Loss function must be defined.')

        def _residual_loss(params, xs):
            resloss = 0
            for i in range(self.residual_loader.numres):
                domain = Residual.split_domain(xs[i])
                for function in self.residual_loader.functions[i]:
                    resloss += self.lossfunc(function(params, *domain),
                                             torch.zeros_like(domain[0]))
            return resloss

        residual_loss = torch.compile(traceable(_residual_loss),
                                      backend=backend,
                                      fullgraph=fullgraph)

        self.residual_loss = residual_loss

    def _train_loop_dataloader(self):
        resloss = 0
        for i, data in enumerate(self.residual_loader.loader):
            xs = torch.vsplit(data.to(self.device, non_blocking=True),
                              self.residual_loader.vsplit)
            resloss += self.residual_loss(self.params, xs)
        resloss /= i + 1.0
        # Backprop
        self.optimizer.zero_grad()
        resloss.backward()
        self.optimizer.step()

        return resloss

    def _train_loop_fullloaded(self):
        resloss = self.residual_loss(self.params, self.residual_loader.xs)

        # Backprop
        self.optimizer.zero_grad()
        resloss.backward()
        self.optimizer.step()

        return resloss

    def fit(self, epochs=1, log_epoch=100, log_progress=10):
        if self.residual_loss is None:
            raise ValueError('Residual Loss must be defined.')
        if self.optimizer is None:
            raise ValueError('Optimizer must be defined.')

        best_loss = float('inf')
        t_ave = 0
        for epoch in range(epochs):
            t0 = thread_time()
            resloss = self.train_loop()
            t_ave = (t_ave * epoch + (thread_time() - t0)) / (epoch + 1)
            if epoch % log_progress == 0:
                _printProgress(epoch, epochs, log_epoch, f'Loss: {resloss:10.6f}',
                               f'{(t_ave*1e3):7.3f}ms/epoch')

            if resloss < best_loss:
                best_loss = resloss
                self.save(epoch, self.name + '_best_model.pt')
        self.save(epoch, self.name + '_final_model.pt')


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
    bar = fill * filledLength + '.' * (length - filledLength)
    print(f'{prefix} [{bar}] [{epoch:>5d}/{epochs:>5d}]% {suffix}',
          flush=True,
          end=printEnd)
