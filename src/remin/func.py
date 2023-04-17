from functools import wraps
from torch import func
from torch import _dynamo as dynamo


def make_sum(f):

    @wraps(f)
    def _sum(*args, **kwargs):
        return f(*args, **kwargs).sum()

    return _sum


def take_col_sum(f, i):

    @wraps(f)
    def _colsum(*args, **kwargs):
        return f(*args, **kwargs)[:, i].sum()

    return _colsum


def traceable(f):
    f = dynamo.allow_in_graph(f)

    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


def functional_call(model):

    def _call(params, *args):
        return func.functional_call(model, dict(params), args)

    return _call


def grad(f, argnum=0):
    return func.grad(make_sum(f), argnums=argnum)


def gradi(f, i, argnum=0):
    return func.grad(take_col_sum(f, i), argnums=argnum)
