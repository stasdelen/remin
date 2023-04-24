import torch
from torch.utils.data import (Dataset, ConcatDataset, BatchSampler, DataLoader)


class Residual:

    def __init__(self, domain, equations, batch_size=None):
        self.domain = domain
        if isinstance(domain, (list, tuple)):
            self.equations = equations
        self.equations = [equations]
        if batch_size is None: self.batch_size = len(domain)
        else: self.batch_size = batch_size

    def to_torch(self, device):
        return torch.from_numpy(self.domain).float().to(device)

    @staticmethod
    def split_domain(domain):
        xs = torch.hsplit(domain, domain.shape[1])
        for x in xs:
            x.requires_grad_()
        return xs


class ResidualDataset(Dataset):

    def __init__(self, domain) -> None:
        self.domain = torch.from_numpy(domain).float()

    def __len__(self):
        return len(self.domain)

    def __getitem__(self, index):
        return self.domain[index]


class ResBatchSampler(BatchSampler):

    def __init__(self, cumulative_sizes, batch_sizes) -> None:
        self.cumul_sizes = cumulative_sizes
        self.batch_sizes = batch_sizes
        self.batch_size = sum(batch_sizes)
        self.ndata = len(batch_sizes)

    def __iter__(self):
        for i in range(self.cumul_sizes[-1] // self.batch_size):
            batch = []
            start_idx = 0
            for idx in range(self.ndata):
                batch_size = self.batch_sizes[idx]
                cumul_size = self.cumul_sizes[idx]
                start = (start_idx + batch_size * i) % cumul_size
                batch += list(range(start, start + batch_size))
                start_idx = cumul_size
            yield batch


class ResidualLoader:

    def __init__(self, residuals, fully_loaded=False, batched=False, **args) -> None:
        if fully_loaded and batched:
            raise ValueError('Fully loaded models can not be batched.')
        self.residuals = residuals
        self.fully_loaded = fully_loaded
        self.batched = batched
        self.numres = len(residuals)
        self.functions = [None] * self.numres
        self.xs = [None] * self.numres
        self.batch_sizes = [None] * self.numres
        self.datasets = [None] * self.numres
        self.batch_sampler = None
        self.args = args

    def instanciate(self, device):
        if self.fully_loaded:
            self.instanciate_fully_loaded(device)
        else:
            self.instanciate_data_loader()

    def instanciate_fully_loaded(self, device):
        for i in range(self.numres):
            self.functions[i] = self.residuals[i].equations
            self.xs[i] = self.residuals[i].to_torch(device)

    def instanciate_data_loader(self):
        for i in range(self.numres):
            self.batch_sizes[i] = self.residuals[i].batch_size
            self.functions[i] = self.residuals[i].equations
            self.datasets[i] = ResidualDataset(self.residuals[i].domain)

        self.dataset = ConcatDataset(self.datasets)
        self.vsplit = [self.batch_sizes[0]]
        for batch_size in self.batch_sizes[1:-1]:
            self.vsplit.append(self.vsplit[-1] + batch_size)
        if self.batched:
            self.batch_sampler = ResBatchSampler(self.dataset.cumulative_sizes,
                                                 self.batch_sizes)
        self.loader = DataLoader(self.dataset,
                                 batch_size=len(self.dataset),
                                 batch_sampler=self.batch_sampler,
                                 **self.args)
