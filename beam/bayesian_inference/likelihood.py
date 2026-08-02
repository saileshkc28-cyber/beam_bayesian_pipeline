"""Likelihood and prior, kept strictly separate.

- Likelihood.log_likelihood(theta) is LIKELIHOOD ONLY and returns a plain Python
  float (numpy >= 2 rejects length-1 arrays in scalar assignment inside SMC_aCS).
  This is the function handed to SMC_aCS: the ERA sampler holds the prior itself
  (pCN proposal preserves it), so adding log_prior here would double-count it.
- Prior.log_pdf(theta) is what plain Metropolis adds on top.

Every forward evaluation is cached (theta -> sensors + displacement field), which
is what lets the per-level SMC VTK statistics be assembled after sampling without
touching SMC_aCS.
"""
import csv
import numpy as np

from era.Distributions.ERADist import ERADist
from era.Distributions.ERANataf import ERANataf

_TINY = np.finfo(float).tiny


def _key(theta):
    return tuple(np.round(np.atleast_1d(theta).astype(float), 12))


class Likelihood:

    def __init__(self, forward_model, measured_data_file, sigma):
        self.fm = forward_model
        self.sigma = float(sigma)
        self.u_hat = self._read_measured(measured_data_file)
        self.cache = {}     # key -> (sensors, displacement_field or None)

    @staticmethod
    def _read_measured(path):
        values = []
        with open(path) as f:
            for row in csv.DictReader(f):
                values.append(float(row["value"]))
        return np.array(values)

    # ------------------------------------------------------------ forward + cache
    def forward(self, theta):
        k = _key(theta)
        if k not in self.cache:
            u = self.fm.evaluate(theta)
            field = self.fm.last_displacement_field
            self.cache[k] = (u, None if field is None else field.copy())
        return self.cache[k][0]

    def field(self, theta):
        self.forward(theta)
        return self.cache[_key(theta)][1]

    # ------------------------------------------------------------ scores
    def residual(self, theta):
        return self.u_hat - self.forward(theta)

    def log_likelihood(self, theta):
        """Likelihood ONLY. Scalar float. This is the SMC_aCS input."""
        r = self.residual(theta)
        return float(-0.5 * float(r @ r) / self.sigma**2)


class Prior:
    """Built from the JSON 'parameters' block. Provides both the ERA object for
    SMC_aCS and marginal log-pdfs for plain Metropolis."""

    def __init__(self, parameter_entries):
        self.names = [e["name"] for e in parameter_entries]
        self.marginals = [
            ERADist(e["prior"]["type"], "PAR", list(e["prior"]["parameters"]))
            for e in parameter_entries
        ]
        self.dim = len(self.marginals)

    def era_object(self):
        """What SMC_aCS receives: a single ERADist for 1D, an ERANataf otherwise."""
        if self.dim == 1:
            return self.marginals[0]
        return ERANataf(self.marginals, np.identity(self.dim))

    def log_pdf(self, theta):
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        lp = 0.0
        for a, dist in zip(theta, self.marginals):
            lp += float(np.log(float(np.atleast_1d(dist.pdf(a))[0]) + _TINY))
        return lp

    def sample(self, rng):
        return np.array([float(np.atleast_1d(d.random(1))[0]) for d in self.marginals])
