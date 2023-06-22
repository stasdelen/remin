import numpy as np
from pyDOE.doe_lhs import lhs


class Domain:

    def __init__(self, n_coll) -> None:
        self.cpoints = None
        self.n_coll = n_coll
        self.time: Time = None

    def __add__(self, rhs):
        return Union(self, rhs)

    def get(self):
        if self.time is None:
            return self.cpoints
        return np.hstack((self.time.cpoints, self.cpoints))


class Time(Domain):

    def __init__(self, t0, t1) -> None:
        super().__init__(n_coll=0)
        self.t0, self.t1 = t0, t1

    def __mul__(self, rhs: Domain):
        rhs.time = self
        return rhs


class Geometry(Domain):

    def _generate(self, time: Time = None) -> None:
        raise NotImplementedError

    def generate(self) -> None:
        if self.time is None:
            self._generate()
        else:
            self._generate(self.time)
        return self

    def _rule(self):
        raise NotImplementedError

    def __sub__(self, rhs):
        return Cut(self, rhs)


class Union(Geometry):

    def __init__(self, lhs: Domain, rhs: Domain) -> None:
        super().__init__(n_coll=0)
        self.lhs = lhs
        self.rhs = rhs

    def _generate(self, time: Time = None) -> None:
        self.lhs._generate(time)
        self.rhs._generate(time)
        if time is None:
            rhs_coll = self.rhs.cpoints
            lhs_coll = self.lhs.cpoints
        else:
            rhs_coll = np.hstack((self.rhs.time.cpoints, self.rhs.cpoints))
            lhs_coll = np.hstack((self.lhs.time.cpoints, self.lhs.cpoints))
        coll = np.vstack((rhs_coll, lhs_coll))
        np.random.shuffle(coll)
        if time is None:
            self.cpoints = coll
        else:
            self.cpoints = coll[:, 1:]
            self.time = Time(time.t0, time.t1)
            self.time.cpoints = coll[:, 0:1]

    def _rule(self):
        return [*self.lhs._rule(), *self.rhs._rule()]


class Cut(Geometry):

    def __init__(self, lhs: Domain, rhs: Domain) -> None:
        super().__init__(n_coll=0)
        self.lhs = lhs
        self.rhs = rhs

    def _generate(self, time=None) -> None:
        self.lhs._generate(time)
        cond = np.logical_not(self.multiple_and(self.lhs.cpoints, self.rhs._rule()))
        self.cpoints = self.lhs.cpoints[cond]
        if time is not None:
            self.time = Time(time.t0, time.t1)
            self.time.cpoints = self.lhs.time.cpoints[cond]

    def _rule(self):
        rhs_rule = lambda cpoints: np.logical_not(
            self.multiple_and(cpoints, self.rhs._rule()))
        return [*self.lhs._rule(), rhs_rule]

    @staticmethod
    def multiple_and(cpoints, rules):
        cond = np.full_like(cpoints[:, 0], True)
        for rule in rules:
            cond = np.logical_and(cond, rule(cpoints))
        return cond


class HyperCube(Geometry):

    def __init__(self, dimensions, n_coll=0) -> None:
        super().__init__(n_coll)
        self.dimensions = dimensions

    def _generate(self, time=None) -> None:
        ndim = len(self.dimensions)
        scale = np.array([dim[1] - dim[0] for dim in self.dimensions])
        shift = np.array([dim[0] for dim in self.dimensions])
        if time is not None:
            colls = lhs(ndim + 1, self.n_coll)
            self.cpoints = colls[:, 1:] * scale + shift
            self.time = Time(time.t0, time.t1)
            self.time.cpoints = colls[:, 0:1] * (time.t1 - time.t0) + time.t0
        else:
            self.cpoints = lhs(ndim, self.n_coll) * scale + shift

    def _rule(self):

        def rule(cpoints):
            cond = np.full_like(cpoints[:, 0], True)
            for i, dim in enumerate(self.dimensions):
                cond = np.logical_and(cond, (cpoints[:, i] >= dim[0]) &
                                      (cpoints[:, i] <= dim[1]))
            return cond

        return [rule]


class Disk(Geometry):

    def __init__(self, center, radius, n_coll=0) -> None:
        super().__init__(n_coll)
        self.center = np.array(center)
        self.radius = radius

    def _generate(self, time=None) -> None:
        if time is not None:
            coll = lhs(3, self.n_coll)
            radius, theta = np.sqrt(coll[:, 1:2]) * self.radius, 2 * np.pi * coll[:,
                                                                                  2:3]
            self.time = Time(time.t0, time.t1)
            self.time.cpoints = coll[:, 0:1] * (time.t1 - time.t0) + time.t0
        else:
            coll = lhs(2, self.n_coll)
            radius, theta = np.sqrt(coll[:, 0:1]) * self.radius, 2 * np.pi * coll[:,
                                                                                  1:2]
        self.cpoints = radius * np.hstack((np.cos(theta), np.sin(theta))) + self.center

    def _rule(self):

        def rule(cpoints):
            return ((cpoints - self.center)**2).sum(axis=1) <= self.radius**2

        return [rule]


class Line(Geometry):

    def __init__(self, pt1, pt2, n_coll=0) -> None:
        super().__init__(n_coll)
        self.pt1, self.pt2 = np.array(pt1), np.array(pt2)

    def _generate(self, time: Time = None) -> None:
        scale = self.pt2 - self.pt1
        if time is None:
            self.cpoints = lhs(1, self.n_coll) * scale + self.pt1
        else:
            coll = lhs(2, self.n_coll)
            self.cpoints = coll[:, 1:] * scale + self.pt1
            self.time = Time(time.t0, time.t1)
            self.time.cpoints = coll[:, 0:1] * (time.t1 - time.t0) + time.t0

    def _rule(self):

        def rule(cpoints):
            cond = np.full_like(cpoints[:, 0], True)
            scale = (cpoints - self.pt1) / (self.pt2 - self.pt1)
            for i in range(len(cpoints)):
                cond[i] = np.allclose(scale[i], scale[i, 0])
            return cond

        return [rule]


class Point(Geometry):

    def __init__(self, pt, n_coll=1) -> None:
        super().__init__(n_coll)
        self.pt = np.array([pt])

    def _generate(self, time: Time = None) -> None:
        if time:
            self.time = Time(time.t0, time.t1)
            self.time.cpoints = lhs(self.n_coll, 1).T * (time.t1 - time.t0) + time.t0
        self.cpoints = np.ones((self.n_coll, 1)) * self.pt

    def _rule(self):

        def rule(cpoints):
            cond = np.full((self.n_coll, ), True)
            return np.allclose(cpoints, self.pt)

        return [rule]


class Ring(Geometry):

    def __init__(self, center, radius, n_coll=0) -> None:
        super().__init__(n_coll)
        self.center = np.array(center)
        self.radius = radius

    def _generate(self, time: Time = None) -> None:
        if time is None:
            theta = 2 * np.pi * lhs(1, self.n_coll)
            self.cpoints = self.radius * np.hstack(
                (np.cos(theta), np.sin(theta))) + self.center
        else:
            coll = lhs(2, self.n_coll)
            theta = 2 * np.pi * coll[:, 1:2]
            self.cpoints = self.radius * np.hstack(
                (np.cos(theta), np.sin(theta))) + self.center
            self.time = Time(time.t0, time.t1)
            self.time.cpoints = coll[:, 0:1] * (time.t1 - time.t0) + time.t0

    def _rule(self):

        def rule(cpoints):
            return np.allclose(((cpoints - self.center)**2).sum(axis=1), self.radius**2)

        return [rule]
