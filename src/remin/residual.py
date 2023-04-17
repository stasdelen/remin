import torch
import numpy as np


class Residual:

    def __init__(self, domain, equation):
        if type(domain) == np.ndarray:
            domain = torch.from_numpy(domain).type(torch.float32)
        self.domain = domain
        self.xs = self.split_domain(domain)
        self.equation = equation

    @staticmethod
    def split_domain(domain: torch.tensor):
        xs = torch.hsplit(domain, domain.shape[1])
        for x in xs:
            x.requires_grad_()
        return xs
