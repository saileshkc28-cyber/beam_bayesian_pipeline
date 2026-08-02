"""Ground truths for the single-zone validation case. NOT part of the pipeline.

1. Closed form: with u(alpha) = u_ref / alpha the likelihood is exactly Gaussian in
   s = 1/alpha:  s_hat = sum(u_hat_i u_ref_i) / sum(u_ref_i^2),
                 Var(s) = sigma^2 / sum(u_ref_i^2)         (flat prior in s).

2. Quadrature: the exact posterior in alpha under the ACTUAL prior of the run,
   integrated on a fine grid. This is the gold standard the samplers must match;
   it also quantifies the (small) Jacobian tilt between (1) and the uniform-in-
   alpha prior. Both die at multi-zone.
"""
import numpy as np


def closed_form_in_s(u_hat, u_ref, sigma):
    s_hat = float(u_hat @ u_ref) / float(u_ref @ u_ref)
    s_std = sigma / np.sqrt(float(u_ref @ u_ref))
    return s_hat, s_std


def quadrature_posterior(likelihood, prior, lo=0.2, hi=2.0, n=200001):
    """Exact 1D posterior of alpha via trapezoid quadrature of L(alpha)*prior(alpha),
    using the (cheap) toy-equivalent likelihood evaluated through the SAME
    Likelihood object, so the residual normalisation is identical."""
    alphas = np.linspace(lo, hi, n)
    # closed relationship u = u_ref/alpha lets the grid be evaluated vectorised
    u_ref = _u_ref_from(likelihood)
    r = likelihood.u_hat[None, :] - u_ref[None, :] / alphas[:, None]
    logL = -0.5 * np.einsum("ij,ij->i", r, r) / likelihood.sigma**2
    logp = logL + np.array([prior.log_pdf([a]) for a in alphas])
    logp -= logp.max()
    p = np.exp(logp)
    Z = np.trapezoid(p, alphas)
    p /= Z
    mean = np.trapezoid(alphas * p, alphas)
    var = np.trapezoid((alphas - mean) ** 2 * p, alphas)
    return alphas, p, mean, np.sqrt(var)


def _u_ref_from(likelihood):
    """Recover u_ref = alpha * u(alpha) from one forward evaluation (exact for the
    linear single-zone map; used only inside this validation module)."""
    a0 = 1.0
    return a0 * likelihood.forward(np.array([a0]))
