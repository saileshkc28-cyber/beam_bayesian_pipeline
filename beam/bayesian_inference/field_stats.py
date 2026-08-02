"""Welford running mean/variance for vector fields (nodal or element arrays).

Storing every sampled field would work for a 22-node beam but not for a realistic
mesh; the accumulator is written the scalable way from the start, as the handoff
requires. reset() is called at the start of each SMC tempering level.
"""
import numpy as np


class FieldStats:

    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self._mean = None
        self._M2 = None

    def update(self, field):
        field = np.asarray(field, dtype=float)
        if self._mean is None:
            self._mean = np.zeros_like(field)
            self._M2 = np.zeros_like(field)
        self.n += 1
        delta = field - self._mean
        self._mean += delta / self.n
        self._M2 += delta * (field - self._mean)

    def mean(self):
        return None if self._mean is None else self._mean.copy()

    def std(self):
        if self._M2 is None or self.n < 2:
            return None if self._M2 is None else np.zeros_like(self._M2)
        return np.sqrt(self._M2 / (self.n - 1))
