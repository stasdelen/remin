import torch
from .func import traceable


class Solver:

    def __init__(self, model, residuals=None) -> None:
        self.model = model
        self.params = dict(model.named_parameters())
        self.equations = [res.equation for res in residuals]
        self.xs = [res.xs for res in residuals]
        self.numres = len(residuals)

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def load(self, filename):
        assert self.model is not None
        state_dict = torch.load(filename)
        self.model.load_state_dict(state_dict)

    def compile(self, optimizer, lossfunc, backend='inductor', fullgraph=False):

        def _residual_loss(params, xs):
            resloss = 0
            for i in range(self.numres):
                domain = xs[i]
                resloss += lossfunc(self.equations[i](params, *domain),
                                    torch.zeros_like(domain[0]))
            return resloss

        residual_loss = torch.compile(traceable(_residual_loss),
                                      backend=backend,
                                      fullgraph=fullgraph)

        # Currently supporting single optimizer
        self.optimizer = optimizer
        self.residual_loss = residual_loss

    def train_loop(self):

        resloss = self.residual_loss(self.params, self.xs)

        # Backprop
        self.optimizer.zero_grad()
        resloss.backward()
        self.optimizer.step()

        return resloss

    def fit(self, epochs=1, log_epoch=100):

        for epoch in range(epochs):

            resloss = self.train_loop()

            if epoch % log_epoch == 0:
                print(f'Loss: {resloss}  [{epoch:>5d}/{epochs:>5d}]')
