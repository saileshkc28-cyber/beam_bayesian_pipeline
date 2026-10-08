"""Pure-numpy core of the adjoint/Laplace Phase 2. No Kratos import on purpose, so
this file can be unit-tested without a Kratos installation.

Contents
--------
Prior               independent uniform / normal priors, bounds, penalty terms
gauss_hermite_rule  probabilists' Gauss-Hermite points (tensor grid over zones)
Evaluation          what one objective evaluation returns
Optimizer           Gauss-Newton (default) or Kratos-style steepest descent + BB step
laplace_posterior   Gaussian posterior N(theta_MAP, H^-1) plus diagnostics
"""
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


# --------------------------------------------------------------------------- prior
class Prior:
    """Independent prior per parameter.

    spec = {"type": "uniform", "parameters": [lower, upper]}
         | {"type": "normal",  "parameters": [mean, std]}

    Uniform: contributes bounds only (no curvature inside the support).
    Normal : contributes 0.5 * ((theta - mean) / std)^2 to the objective.
    """

    def __init__(self, specs, names):
        self.names = list(names)
        n = len(specs)
        self.lower = np.full(n, -np.inf)
        self.upper = np.full(n, np.inf)
        self.mean = np.zeros(n)
        self.precision = np.zeros(n)
        self.types = []
        for i, spec in enumerate(specs):
            kind = spec["type"].lower()
            p = [float(v) for v in spec["parameters"]]
            if kind == "uniform":
                if not p[0] < p[1]:
                    raise ValueError(f"uniform prior of '{names[i]}' needs lower < upper, got {p}")
                self.lower[i], self.upper[i] = p[0], p[1]
            elif kind in ("normal", "gaussian"):
                if p[1] <= 0.0:
                    raise ValueError(f"normal prior of '{names[i]}' needs std > 0, got {p}")
                self.mean[i], self.precision[i] = p[0], 1.0 / p[1] ** 2
            else:
                raise ValueError(f"unsupported prior type '{kind}' for '{names[i]}' "
                                 "(use 'uniform' or 'normal')")
            self.types.append(kind)

    def value(self, theta):
        d = np.asarray(theta, float) - self.mean
        return 0.5 * float(np.sum(self.precision * d * d))

    def gradient(self, theta):
        return self.precision * (np.asarray(theta, float) - self.mean)

    def hessian(self):
        return np.diag(self.precision)

    def project(self, theta):
        return np.clip(np.asarray(theta, float), self.lower, self.upper)

    def in_support(self, samples):
        s = np.atleast_2d(samples)
        return np.all((s >= self.lower) & (s <= self.upper), axis=1)

    def distance_to_bound_in_sd(self, theta, sd):
        """How many posterior sd the MAP sits away from the nearest finite bound."""
        theta = np.asarray(theta, float)
        with np.errstate(invalid="ignore", divide="ignore"):
            lo = np.where(np.isfinite(self.lower), (theta - self.lower) / sd, np.inf)
            hi = np.where(np.isfinite(self.upper), (self.upper - theta) / sd, np.inf)
        return np.minimum(lo, hi)


# ------------------------------------------------------------ Gauss-Hermite rule
def gauss_hermite_rule(order, n_dims):
    """Probabilists' Gauss-Hermite rule for a standard normal, tensor grid in n_dims.

    Returns Z (n_points x n_dims) and weights W (n_points,), sum(W) = 1, so that
    E[f(xi)] ~= sum_q W_q f(Z_q) for xi ~ N(0, I).
    order = 3 gives z = (-sqrt(3), 0, +sqrt(3)), w = (1/6, 2/3, 1/6).
    """
    x, w = np.polynomial.hermite_e.hermegauss(int(order))
    w = w / w.sum()
    grids = np.meshgrid(*([x] * n_dims), indexing="ij")
    wgrids = np.meshgrid(*([w] * n_dims), indexing="ij")
    Z = np.stack([g.ravel() for g in grids], axis=1)
    W = np.prod(np.stack([g.ravel() for g in wgrids], axis=1), axis=1)
    return Z, W


# ------------------------------------------------------------------- evaluation
@dataclass
class Evaluation:
    """One evaluation of the objective J(theta) = data misfit + prior penalty.

    J : objective value
    g : dJ/dtheta
    H : Gauss-Newton Hessian R^T R + prior precision (None if not computed)
    info : free-form diagnostics (predictions, cross-checks, ...)
    """
    theta: np.ndarray
    J: float
    g: np.ndarray
    H: Optional[np.ndarray] = None
    info: dict = field(default_factory=dict)


# -------------------------------------------------------------------- optimizer
class Optimizer:
    """Gradient-based minimisation of J, mirroring the CVaR/Kratos update loop.

    "gauss_newton"        : step = -H^-1 g with Armijo backtracking. Needs H every
                            iteration (one adjoint per sensor). Converges in a few
                            iterations for a handful of parameters.
    "steepest_descent_bb" : the update of Kratos' AlgorithmSteepestDescent with the
                            BB_step line search (same formulas as opt_line_search.py):
                            direction = -g, first step init_step/||d||, then
                            |d_prev.d_prev / d_prev.y| capped at max_step/||d||.
                            Uses the gradient only.
    """

    DEFAULTS = {
        "type": "gauss_newton",
        "max_iterations": 50,
        "decrement_tolerance": 1e-10,
        "step_tolerance": 1e-10,
        "objective_rel_tolerance": 1e-12,
        "max_backtracking": 20,
        "bb_settings": {"init_step": 0.01, "max_step": 0.1, "gradient_scaling": "l2_norm"},
    }

    def __init__(self, evaluate: Callable, prior: Prior, settings: dict, log=print):
        self.evaluate = evaluate
        self.prior = prior
        self.s = settings
        self.log = log
        if self.s["type"] not in ("gauss_newton", "steepest_descent_bb"):
            raise ValueError(f"unknown optimizer type '{self.s['type']}'")

    # -- helpers
    @staticmethod
    def newton_step(g, H):
        try:
            return -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return -np.linalg.lstsq(H, g, rcond=None)[0]

    def _bb_norm(self, d):
        scaling = self.s["bb_settings"]["gradient_scaling"]
        if scaling == "l2_norm":
            n = float(np.linalg.norm(d))
        elif scaling == "inf_norm":
            n = float(np.max(np.abs(d)))
        elif scaling == "none":
            n = 1.0
        else:
            raise ValueError(f"unknown gradient_scaling '{scaling}'")
        return 1.0 if math.isclose(n, 0.0, abs_tol=1e-16) else n

    def _row(self, it, ev, step_norm, decrement, t):
        return {"iteration": it, "theta": ev.theta.copy(), "J": ev.J,
                "grad_norm": float(np.linalg.norm(ev.g)), "decrement": decrement,
                "step_norm": step_norm, "step_length": t}

    # -- main loop
    def run(self, theta0):
        gn = self.s["type"] == "gauss_newton"
        theta = self.prior.project(theta0)
        ev = self.evaluate(theta, gn)
        if ev is None:
            raise RuntimeError(f"objective is not defined at the initial guess {theta}")
        history = [self._row(0, ev, 0.0, float("nan"), float("nan"))]
        converged, reason = False, "max_iterations"
        prev_dir = prev_update = None

        for it in range(1, int(self.s["max_iterations"]) + 1):
            if gn:
                d = self.newton_step(ev.g, ev.H)
                slope = float(ev.g @ d)
                decrement = -slope
                if decrement < self.s["decrement_tolerance"]:
                    converged, reason = True, "newton_decrement"
                    history[-1]["decrement"] = decrement
                    break
                t, accepted = 1.0, None
                for _ in range(int(self.s["max_backtracking"]) + 1):
                    trial = self.prior.project(theta + t * d)
                    ev_t = self.evaluate(trial, True)
                    if ev_t is not None and ev_t.J <= ev.J + 1e-4 * t * slope:
                        accepted = ev_t
                        break
                    t *= 0.5
                if accepted is None:
                    reason = "line_search_stalled"
                    self.log(f"  iteration {it}: no decrease along the Gauss-Newton "
                             "direction; stopping (optimum reached to working precision).")
                    converged = decrement < 1e3 * self.s["decrement_tolerance"]
                    break
            else:
                d = -ev.g
                norm = self._bb_norm(d)
                bb = self.s["bb_settings"]
                if prev_dir is None:
                    eta = bb["init_step"] / norm
                else:
                    y = prev_dir - d
                    dy = float(prev_update @ y)
                    dd = float(prev_update @ prev_update)
                    eta = bb["max_step"] / norm if math.isclose(dy, 0.0, abs_tol=1e-300) \
                        else abs(dd / dy)
                    eta = min(eta, bb["max_step"] / norm)
                accepted, t = None, 1.0
                for _ in range(30):   # only shrinks if a trial point is infeasible
                    trial = self.prior.project(theta + t * eta * d)
                    ev_t = self.evaluate(trial, False)
                    if ev_t is not None:
                        accepted = ev_t
                        break
                    t *= 0.5
                if accepted is None:
                    reason = "infeasible_step"
                    break
                decrement = float("nan")
                prev_dir, prev_update = d, trial - theta
                t = t * eta

            step = accepted.theta - theta
            rel_change = abs(accepted.J - ev.J) / max(abs(ev.J), 1e-300)
            theta, ev = accepted.theta, accepted
            step_norm = float(np.max(np.abs(step)) / max(1.0, float(np.max(np.abs(theta)))))
            history.append(self._row(it, ev, step_norm, decrement, t))
            self.log(f"  iteration {it:3d}: J = {ev.J:.10e}  |g| = {np.linalg.norm(ev.g):.3e}"
                     f"  step = {step_norm:.3e}  theta = {np.array2string(theta, precision=8)}")

            if step_norm < self.s["step_tolerance"]:
                converged, reason = True, "step_tolerance"
                break
            if not gn and rel_change < self.s["objective_rel_tolerance"]:
                converged, reason = True, "objective_rel_tolerance"
                break

        if ev.H is None:   # steepest descent never needed H; build it once for Laplace
            ev = self.evaluate(theta, True)
        return ev, history, converged, reason


# ------------------------------------------------------------ Laplace posterior
def laplace_posterior(ev: Evaluation, prior: Prior, n_samples=100000, seed=0):
    """Gaussian approximation N(theta_MAP, H^-1) at the optimum.

    H is the Gauss-Newton Hessian (sensitivities from adjoints). Diagnostics:
    gradient norm and Newton decrement at the MAP (should be ~0 for Laplace to
    be valid), eigenvalues of H (a tiny one = a direction the data cannot see),
    distance of the MAP to prior bounds in posterior sd.
    """
    H = 0.5 * (ev.H + ev.H.T)
    w, V = np.linalg.eigh(H)
    warnings = []
    if w.min() <= 0.0:
        warnings.append("Hessian not positive definite: at least one direction is not "
                        "identified by the data; its variance is set by pseudo-inverse "
                        "and is not meaningful.")
        w_inv = np.where(w > w.max() * 1e-14, 1.0 / np.where(w > 0, w, 1.0), 0.0)
    else:
        w_inv = 1.0 / w
    cov = (V * w_inv) @ V.T
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(sd, sd)
    cond = float(w.max() / w.min()) if w.min() > 0 else float("inf")
    if cond > 1e10:
        warnings.append(f"Hessian condition number {cond:.2e}: parameters are close to "
                        "non-identifiable (see the eigenvector of the smallest eigenvalue).")

    decrement = float(ev.g @ np.linalg.lstsq(H, ev.g, rcond=None)[0])
    if decrement > 1e-6:
        warnings.append(f"Newton decrement at the optimum is {decrement:.2e} (> 1e-6): the "
                        "point is not a converged optimum, so the Laplace approximation "
                        "is not centred correctly.")

    bound_dist = prior.distance_to_bound_in_sd(ev.theta, np.where(sd > 0, sd, np.inf))
    for name, dist in zip(prior.names, bound_dist):
        if dist < 3.0:
            warnings.append(f"'{name}' lies {dist:.2f} posterior sd from a prior bound: the "
                            "Gaussian approximation is truncated there.")

    rng = np.random.default_rng(seed)
    L = V * np.sqrt(w_inv)
    samples = ev.theta + rng.standard_normal((int(n_samples), len(ev.theta))) @ L.T
    inside = prior.in_support(samples)

    return {
        "map": ev.theta.copy(), "covariance": cov, "std": sd, "correlation": corr,
        "hessian": H, "hessian_eigenvalues": w, "hessian_eigenvectors": V,
        "condition_number": cond, "gradient_norm": float(np.linalg.norm(ev.g)),
        "newton_decrement": decrement, "samples": samples,
        "fraction_outside_prior": float(1.0 - inside.mean()), "warnings": warnings,
    }
