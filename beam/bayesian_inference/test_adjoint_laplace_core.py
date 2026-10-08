"""Kratos-free checks of the adjoint/Laplace math.

Run:  python test_adjoint_laplace_core.py

The mock forward model u_k = sum_i C[k, i] / alpha_i has the same structure as a
linear-elastic structure with zone-wise Young's modulus (compliance ~ 1/E), and its
sensitivities are known in closed form, so every step can be checked exactly.
Finite differences appear ONLY here, as an independent test of the analytic
chain rule; the Phase 2 method itself never uses them.
"""
import numpy as np

from adjoint_laplace_core import Prior, Optimizer, gauss_hermite_rule, laplace_posterior
from expectation_evaluator import UMeanObjective, GaussHermiteMomentObjective


class MockForward:
    def __init__(self, C):
        self.C = np.asarray(C, float)
        self.sensor_weights = np.ones(self.C.shape[0])
        self.alpha = None
        self.n_primal = self.n_adjoint = 0

    def Evaluate(self, alpha):
        self.alpha = np.asarray(alpha, float)
        self.n_primal += 1
        return self.C @ (1.0 / self.alpha)

    def SensorJacobian(self):
        self.n_adjoint += self.C.shape[0]
        return -self.C / self.alpha[None, :] ** 2

    def MisfitValueAndGradient(self, d):
        self.n_adjoint += 1
        e = self.C @ (1.0 / self.alpha) - d
        return 0.5 * float(e @ e), self.SensorJacobian_noCount().T @ e

    def SensorJacobian_noCount(self):
        return -self.C / self.alpha[None, :] ** 2


def quiet(*_):
    pass


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        raise SystemExit(1)


def test_rule():
    Z, W = gauss_hermite_rule(3, 1)
    check("3-point rule", np.allclose(sorted(Z[:, 0]), [-np.sqrt(3), 0, np.sqrt(3)])
          and np.allclose(sorted(W), [1 / 6, 1 / 6, 2 / 3]))
    Z, W = gauss_hermite_rule(3, 2)
    check("tensor rule reproduces N(0, I) moments",
          np.isclose(W.sum(), 1) and np.allclose(W @ Z, 0) and np.allclose((W[:, None] * Z).T @ Z, np.eye(2)))


def test_u_mean(opt_type):
    C = np.array([[2.0e-6], [1.0e-6]])
    fwd = MockForward(C)
    alpha_true, sigma = np.array([0.8]), 2e-8
    d = C @ (1 / alpha_true)
    prior = Prior([{"type": "uniform", "parameters": [0.2, 2.0]}], ["alpha_1"])
    obj = UMeanObjective(fwd, d, sigma, prior, log=quiet)
    settings = dict(Optimizer.DEFAULTS, type=opt_type,
                    max_iterations=50 if opt_type == "gauss_newton" else 3000)
    ev, hist, conv, reason = Optimizer(obj.evaluate, prior, settings, log=quiet).run([1.0])
    lap = laplace_posterior(ev, prior, 20000, 1)
    sd_exact = sigma / np.linalg.norm(C[:, 0] / alpha_true[0] ** 2)
    check(f"u_mean {opt_type}: MAP", abs(ev.theta[0] - 0.8) < 1e-7,
          f"(alpha = {ev.theta[0]:.10f}, {len(hist) - 1} iterations, {reason})")
    check(f"u_mean {opt_type}: Laplace sd", np.isclose(lap["std"][0], sd_exact, rtol=1e-6),
          f"({lap['std'][0]:.6e} vs exact {sd_exact:.6e})")
    check(f"u_mean {opt_type}: Kratos-J gradient == per-sensor gradient",
          obj.max_gradient_mismatch < 1e-10, f"({obj.max_gradient_mismatch:.1e})")


def fd_jacobian_check():
    """The analytic d(residual)/d(mu, s) against central differences (test only)."""
    C = np.array([[2.0e-6, 0.5e-6], [1.0e-6, 1.5e-6], [0.3e-6, 0.8e-6]])
    fwd = MockForward(C)
    data = {"mean": np.array([2.4e-6, 2.4e-6, 1.1e-6]), "std": np.array([2e-7, 2e-7, 1e-7]),
            "n_samples": 1000, "sensor_noise_sigma": 3e-8}
    prior = Prior([{"type": "uniform", "parameters": [0.2, 2]}] * 2 +
                  [{"type": "uniform", "parameters": [1e-4, 0.5]}] * 2, ["m1", "m2", "s1", "s2"])
    obj = GaussHermiteMomentObjective(fwd, data, prior, 2, log=quiet)
    theta = np.array([0.95, 1.1, 0.08, 0.12])
    pred = obj.predict(theta)
    r0, R = obj.residual(pred)
    R_fd = np.zeros_like(R)
    for j in range(4):
        h = 1e-6 * max(1, abs(theta[j]))
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h
        tm[j] -= h
        R_fd[:, j] = (obj.residual(obj.predict(tp, False), False)[0]
                      - obj.residual(obj.predict(tm, False), False)[0]) / (2 * h)
    err = np.max(np.abs(R - R_fd)) / np.max(np.abs(R))
    check("GH chain rule vs central differences", err < 1e-6, f"(rel err {err:.1e})")


def population_moments(C, mu, s, sigma_noise, npts=200):
    """'Phase 1' statistics by a very fine quadrature (stands in for 1000 samples)."""
    Z, W = gauss_hermite_rule(npts, 1)
    alphas = mu + s * Z[:, 0]
    keep = alphas > 0
    U = np.array([C @ (1 / np.array([a])) for a in alphas[keep]])
    Wk = W[keep] / W[keep].sum()
    m = Wk @ U
    v = np.sqrt(Wk @ (U - m) ** 2 + sigma_noise**2)
    return m, v


def test_gauss_hermite(match, expect_ok):
    C = np.array([[1.894e-6]])
    fwd = MockForward(C)
    mu_t, s_t, noise, n = 1.0, 0.1, 3.788e-8, 983
    m, v = population_moments(C, mu_t, s_t, noise)
    data = {"mean": m, "std": v, "n_samples": n, "sensor_noise_sigma": noise}
    prior = Prior([{"type": "uniform", "parameters": [0.2, 2.0]},
                   {"type": "uniform", "parameters": [1e-4, 0.5]}], ["mu_1", "sigma_1"])
    obj = GaussHermiteMomentObjective(fwd, data, prior, 1, match=match, log=quiet)
    settings = dict(Optimizer.DEFAULTS)
    ev, hist, conv, reason = Optimizer(obj.evaluate, prior, settings, log=quiet).run([1.05, 0.05])
    lap = laplace_posterior(ev, prior, 20000, 2)
    tag = "+".join(match)
    if expect_ok:
        check(f"GH [{tag}]: recovers mu", abs(ev.theta[0] - mu_t) < 3e-3,
              f"(mu = {ev.theta[0]:.5f} +/- {lap['std'][0]:.5f})")
        check(f"GH [{tag}]: recovers sigma", abs(ev.theta[1] - s_t) < 3e-3,
              f"(sigma = {ev.theta[1]:.5f} +/- {lap['std'][1]:.5f}, "
              f"{fwd.n_primal} primal / {fwd.n_adjoint} adjoint solves)")
    else:
        check(f"GH [{tag}]: flags non-identifiability", len(lap["warnings"]) > 0
              or lap["condition_number"] > 1e8,
              f"(cond = {lap['condition_number']:.2e}, corr = {lap['correlation'][0, 1]:+.4f})")


if __name__ == "__main__":
    test_rule()
    test_u_mean("gauss_newton")
    test_u_mean("steepest_descent_bb")
    fd_jacobian_check()
    test_gauss_hermite(("mean", "std"), True)
    test_gauss_hermite(("mean",), False)
    print("all checks passed")
