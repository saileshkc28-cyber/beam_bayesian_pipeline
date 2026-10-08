"""Objective functions for the adjoint/Laplace Phase 2. Pure numpy, no Kratos import.

Both objectives talk to a "forward" object with this interface (implemented with
Kratos in adjoint_sensitivity_model.AdjointSensitivityModel, and with a closed-form
mock in test_adjoint_laplace_core.py):

    Evaluate(alpha)            -> u        primal solve, predicted sensor vector
    MisfitValueAndGradient(d)  -> Jk, gk   Kratos J = sum 0.5 w (u - d)^2 and dJ/dalpha
                                           (one adjoint), at the last primal state
    SensorJacobian()           -> S        du_k/dalpha_i (one adjoint per sensor),
                                           at the last primal state
    sensor_weights             -> w        weights used inside the Kratos J

UMeanObjective              stage 1: one alpha per zone, fitted to u_mean
GaussHermiteMomentObjective stage 2: population mean and sd per zone, fitted to the
                            mean (and sd) of the Phase 1 responses via Gauss-Hermite
"""
import numpy as np

from adjoint_laplace_core import Evaluation, gauss_hermite_rule


class UMeanObjective:
    """J(alpha) = 0.5 * sum_k (u_k(alpha) - d_k)^2 / sigma^2 + prior penalty.

    The data part is computed exactly as in the CVaR/DamageDetectionResponse
    setup: Kratos MeasurementResidualResponseFunction (p = 1) for the value and one
    adjoint solve for the gradient. The per-sensor adjoints (S) are only needed for
    the Gauss-Newton Hessian, and are used to cross-check the Kratos gradient.
    """

    def __init__(self, forward, d, sigma, prior, log=print):
        self.fwd = forward
        self.d = np.asarray(d, float)
        self.sigma = float(sigma)
        self.prior = prior
        self.log = log
        w = np.asarray(forward.sensor_weights, float)
        if not np.allclose(w, 1.0):
            raise ValueError("sensor weights in sensor_data.json must all be 1.0 for the "
                             f"iid Gaussian likelihood (found {w}); remove or reset them")
        self.max_gradient_mismatch = 0.0

    def names(self, zone_names):
        return list(zone_names)

    def evaluate(self, theta, need_hessian):
        alpha = np.asarray(theta, float)
        u = np.asarray(self.fwd.Evaluate(alpha), float)
        Jk, gk = self.fwd.MisfitValueAndGradient(self.d)
        J = Jk / self.sigma**2 + self.prior.value(alpha)
        g = np.asarray(gk, float) / self.sigma**2 + self.prior.gradient(alpha)
        info = {"u": u, "residual_over_sigma": (u - self.d) / self.sigma,
                "J_data_kratos": Jk / self.sigma**2}
        H = None
        if need_hessian:
            S = np.asarray(self.fwd.SensorJacobian(), float)
            r = (u - self.d) / self.sigma
            R = S / self.sigma
            H = R.T @ R + self.prior.hessian()
            # cross-check: Kratos J and its adjoint gradient against the same
            # quantities rebuilt from the per-sensor adjoints
            g_py = R.T @ r
            g_k = np.asarray(gk, float) / self.sigma**2
            scale = max(np.max(np.abs(g_py)), np.max(np.abs(g_k)), 1e-300)
            mismatch = float(np.max(np.abs(g_py - g_k)) / scale)
            self.max_gradient_mismatch = max(self.max_gradient_mismatch, mismatch)
            info.update({"S": S, "J_data_python": 0.5 * float(r @ r),
                         "gradient_cross_check": mismatch})
        return Evaluation(theta=alpha.copy(), J=float(J), g=g, H=H, info=info)


class GaussHermiteMomentObjective:
    """Population inference from the Phase 1 response statistics.

    theta = (mu_1..mu_z, s_1..s_z): mean and sd of alpha per zone.
    For every Gauss-Hermite point q (tensor grid over zones):
        alpha_q = mu + s * z_q                  -> Kratos solve -> u_q
        adjoint per sensor at alpha_q           -> S_q = du/dalpha at alpha_q
    Expectation (the CVaR-style weighted combination of realizations):
        u_bar = sum_q W_q u_q
        var   = sum_q W_q (u_q - u_bar)^2 + sigma_noise^2   (the data u_hat include noise)
    Residuals (standard errors of the sample mean and sample sd of n responses):
        r_mean = (u_bar - m) / (v / sqrt(n))
        r_sd   = (sqrt(var) - v) / (v / sqrt(2 (n - 1)))
    J = 0.5 |r|^2 + prior penalty; exact derivatives via the chain rule through S_q.

    Unlike CVaR (which averages the LOSSES of the realizations), here the
    PREDICTIONS are averaged first and then compared with the data, so every
    point's adjoint must be weighted by the combined residual. That is why the
    per-sensor sensitivities S_q are used instead of one misfit adjoint per point.
    """

    def __init__(self, forward, data, prior, n_zones, order=3, match=("mean", "std"),
                 min_node_alpha=0.05, log=print):
        self.fwd = forward
        self.prior = prior
        self.nz = int(n_zones)
        self.Z, self.W = gauss_hermite_rule(order, self.nz)
        self.match = tuple(match)
        if not self.match or any(m not in ("mean", "std") for m in self.match):
            raise ValueError(f"'match' must be a non-empty subset of ['mean', 'std'], got {match}")
        self.m = np.asarray(data["mean"], float)
        self.v = np.asarray(data["std"], float)
        self.n = float(data["n_samples"])
        self.sigma_noise = float(data["sensor_noise_sigma"])
        self.se_m = self.v / np.sqrt(self.n)
        self.se_v = self.v / np.sqrt(2.0 * (self.n - 1.0))
        if np.any(self.v <= 0.0):
            raise ValueError("the sd of the Phase 1 responses must be > 0 for every sensor")
        if np.any(self.v <= self.sigma_noise):
            log("WARNING: the response sd is not larger than the sensor noise at some "
                "sensor; the population sd cannot be separated from noise there.")
        self.min_node_alpha = float(min_node_alpha)
        self.log = log

    def names(self, zone_names):
        return [f"mu_{z}" for z in zone_names] + [f"sigma_{z}" for z in zone_names]

    def node_alphas(self, theta):
        mu, s = theta[:self.nz], theta[self.nz:]
        return mu[None, :] + self.Z * s[None, :]

    def predict(self, theta, with_jacobian=True):
        """Weighted combination of the Gauss-Hermite realizations (one Kratos solve,
        plus one adjoint per sensor, per point)."""
        A = self.node_alphas(theta)
        U, D = [], []
        for q, alpha_q in enumerate(A):
            U.append(np.asarray(self.fwd.Evaluate(alpha_q), float))
            if with_jacobian:
                S_q = np.asarray(self.fwd.SensorJacobian(), float)
                # d u_q / d(mu, s) = [S_q, S_q * z_q]
                D.append(np.hstack([S_q, S_q * self.Z[q][None, :]]))
        U = np.array(U)
        u_bar = self.W @ U
        dev = U - u_bar
        var = self.W @ (dev * dev) + self.sigma_noise**2
        sd = np.sqrt(var)
        out = {"alphas": A, "U": U, "mean": u_bar, "std": sd}
        if with_jacobian:
            D = np.array(D)                                         # (Q, ns, 2nz)
            dmean = np.einsum("q,qkp->kp", self.W, D)
            dvar = 2.0 * np.einsum("q,qk,qkp->kp", self.W, dev, D)  # sum_q W dev = 0
            out["dmean"] = dmean
            out["dstd"] = dvar / (2.0 * sd[:, None])
        return out

    def residual(self, pred, with_jacobian=True):
        r, R = [], []
        if "mean" in self.match:
            r.append((pred["mean"] - self.m) / self.se_m)
            if with_jacobian:
                R.append(pred["dmean"] / self.se_m[:, None])
        if "std" in self.match:
            r.append((pred["std"] - self.v) / self.se_v)
            if with_jacobian:
                R.append(pred["dstd"] / self.se_v[:, None])
        return np.concatenate(r), (np.vstack(R) if with_jacobian else None)

    def evaluate(self, theta, need_hessian=True):
        theta = np.asarray(theta, float)
        if np.any(self.node_alphas(theta) <= self.min_node_alpha):
            return None     # a Gauss-Hermite point would have (near) zero stiffness
        pred = self.predict(theta, True)     # the gradient needs S_q in any case
        r, R = self.residual(pred, True)
        J = 0.5 * float(r @ r) + self.prior.value(theta)
        g = R.T @ r + self.prior.gradient(theta)
        H = R.T @ R + self.prior.hessian()
        info = {"alphas": pred["alphas"], "U": pred["U"], "pred_mean": pred["mean"],
                "pred_std": pred["std"], "residual": r}
        return Evaluation(theta=theta.copy(), J=J, g=g, H=H, info=info)
