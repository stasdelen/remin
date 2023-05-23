import numpy as np
from pyDOE.doe_lhs import lhs


def HyperCube(dimensions, n_coll):
    ndim = len(dimensions)
    scale = np.array([dim[1] - dim[0] for dim in dimensions])
    shift = np.array([dim[0] for dim in dimensions])
    return lhs(ndim, n_coll) * scale + shift


def Circle(center, radius, n_coll):
    center = np.array(center)
    coll = lhs(2, n_coll)
    radius, theta = coll[0], 2 * np.pi * coll[1]
    return radius * np.hstack((np.cos(theta), np.sin(theta))) + center


def Line(pt1, pt2, n_coll):
    ndim = len(pt1)
    scale = np.array([x2 - x1 for x1, x2 in zip(pt1, pt2)])
    shift = np.array(pt1)
    return lhs(ndim, n_coll) * scale + shift


def Point(pt):
    return np.array([pt])


def Ring(center, radius, n_coll):
    center = np.array(center)
    theta = 2 * np.pi * lhs(1, n_coll)
    return radius * np.hstack((np.cos(theta), np.sin(theta))) + center


def cut(domain_from, constraints):
    if not isinstance(constraints, list):
        constraints = [constraints]
    cond = np.full_like(domain_from[:, 0], True)
    for constraint in constraints:
        cond = np.logical_and(cond, constraint(domain_from))
    return domain_from[cond]


def union(*args):
    return np.vstack(args)


def shuffle(domain):
    np.random.shuffle(domain)
    return domain
