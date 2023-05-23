import torch
from torch.utils.data import (Dataset, ConcatDataset, BatchSampler, DataLoader)


class Residual:

    def __init__(self, domain, equations, weight=1.0, batch_size=None):
        self.domain = domain
        self.weight = weight
        if isinstance(equations, (list, tuple)):
            self.equations = equations
        else:
            self.equations = [equations]
        if batch_size is None:
            self.batch_size = len(domain)
        else:
            self.batch_size = batch_size

    def to_torch(self, device):
        return torch.from_numpy(self.domain.astype('float32')).to(device)

    @staticmethod
    def split_domain(domain):
        if domain.shape[1] == 1:
            domain.requires_grad_()
            return (domain, )
        xs = torch.hsplit(domain, domain.shape[1])
        for x in xs:
            x.requires_grad_()
        return xs


class ResidualDataset(Dataset):

    def __init__(self, domain) -> None:
        self.domain = torch.from_numpy(domain.astype('float32'))

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
        self.n_batches = self.cumul_sizes[0] // self.batch_sizes[0]
        for i in range(1, self.ndata):
            n = (self.cumul_sizes[i] - self.cumul_sizes[i - 1]) // self.batch_sizes[i]
            if n > self.n_batches:
                self.n_batches = n

        self.batch = [None] * self.batch_size
        self.batch_shift = [None] * self.batch_size
        start_idx, batch_idx = 0, 0
        for i in range(self.ndata):
            batch_size = self.batch_sizes[i]
            cumul_size = self.cumul_sizes[i]
            prev_cumul_size = self.cumul_sizes[i - 1] if i > 0 else 0
            for j in range(batch_size):
                self.batch_shift[batch_idx + j] = (batch_size, cumul_size,
                                                   prev_cumul_size)
                self.batch[batch_idx + j] = start_idx + j
            start_idx = cumul_size
            batch_idx += batch_size

    def __iter__(self):
        for _ in range(self.n_batches):
            yield self.batch
            for j in range(self.batch_size):
                batch_size, cumul_size, prev_cumul_size = self.batch_shift[j]
                self.batch[j] = self.batch[j] + batch_size
                if self.batch[j] >= cumul_size:
                    self.batch[j] -= cumul_size - prev_cumul_size


class Loader:

    def __init__(self, residuals, **kwargs) -> None:
        self.residuals = residuals
        self.n_res = len(residuals)
        self.functions = [None] * self.n_res
        self.weights = [None] * self.n_res
        self.kwargs = kwargs

    def instanciate(self, device):
        raise NotImplementedError('Base class cannot be instanciated.')


class FullLoader(Loader):

    def __init__(self, residuals, **kwargs) -> None:
        super().__init__(residuals, **kwargs)
        self.xs = [None] * self.n_res

    def instanciate(self, device):
        for i in range(self.n_res):
            self.functions[i] = self.residuals[i].equations
            self.xs[i] = self.residuals[i].to_torch(device)
            self.weights[i] = self.residuals[i].weight


class BatchLoader(Loader):

    def __init__(self, residuals, batched, **kwargs) -> None:
        super().__init__(residuals, **kwargs)
        self.batch_sizes = [None] * self.n_res
        self.datasets = [None] * self.n_res
        self.batch_sampler = None
        self.batched = batched

        for i in range(self.n_res):
            self.batch_sizes[i] = self.residuals[i].batch_size
            self.functions[i] = self.residuals[i].equations
            self.weights[i] = self.residuals[i].weight
            self.datasets[i] = ResidualDataset(self.residuals[i].domain)

        self.dataset = ConcatDataset(self.datasets)
        self.dataset_size = len(self.dataset)
        self.vsplit = [self.batch_sizes[0]]

        for batch_size in self.batch_sizes[1:-1]:
            self.vsplit.append(self.vsplit[-1] + batch_size)
        if self.batched:
            self.batch_sampler = ResBatchSampler(self.dataset.cumulative_sizes,
                                                 self.batch_sizes)
            self.loader = DataLoader(self.dataset,
                                     batch_sampler=self.batch_sampler,
                                     **self.kwargs)
        else:
            self.loader = DataLoader(self.dataset,
                                     batch_size=self.dataset_size,
                                     **self.kwargs)

    def instanciate(self, device):
        pass


def make_loader(residuals, fully_loaded=False, batched=False, **kwargs):
    if fully_loaded and batched:
        raise ValueError('Data can not be fully loaded and batched at the same time.')
    if fully_loaded:
        return FullLoader(residuals, **kwargs)
    return BatchLoader(residuals, batched, **kwargs)
