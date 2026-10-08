"""Phase 2 by adjoint optimisation + Laplace approximation.

    parameter guess -> Kratos primal -> predicted u -> J -> adjoint gradient
    -> parameter update -> ... -> converged optimum (MAP) -> Gauss-Newton Hessian
    from adjoint sensor sensitivities -> Laplace posterior N(MAP, H^-1)

mode "u_mean"        : one alpha per zone fitted to the collapsed Phase 1 mean
                       response (value column of measured_data_collapsed.csv)
mode "gauss_hermite" : population mean and sd of alpha per zone, fitted to the
                       mean and sd of the Phase 1 responses; each evaluation runs
                       one Kratos primal (+ adjoints) per Gauss-Hermite point and
                       combines them with the quadrature weights

Phase 2 reads displacements only. The Phase 1 alpha values are never used for
inference; the optional validation_reference_file is read after the run, only to
report the error of the result.
"""
import copy
import csv
import json
import os

import numpy as np
import KratosMultiphysics as Kratos

from adjoint_laplace_core import Prior, Optimizer, laplace_posterior
from expectation_evaluator import UMeanObjective, GaussHermiteMomentObjective

PREFIX = "AdjointLaplace"

DEFAULTS = {
    "mode": "u_mean",
    "adjoint_parameters_file": "AdjointParametersBayes.json",
    "element_name": "ShellThinElement3D3N",
    "measured_data_file": "../damaged_system/phase1_distribution_runs/measured_data_collapsed.csv",
    "noise_model_file": "../damaged_system/phase1_distribution_runs/noise_model_collapsed.json",
    "u_mean": {
        "initial_guess": [1.0],
        "sigma_key": "sigma",
        "sigma_override": 0.0,
    },
    "gauss_hermite": {
        "order": 3,
        "initial_mean": [1.0],
        "initial_std": [0.05],
        "std_prior": {"type": "uniform", "parameters": [1e-4, 0.5]},
        "match": ["mean", "std"],
        "sensor_noise_sigma_key": "sigma_options.sensor_noise",
        "min_node_alpha": 0.05,
    },
    "optimizer": copy.deepcopy(Optimizer.DEFAULTS),
    "checks": {
        "run_startup_checks": True,
        "homogeneity_tolerance": 1e-4,
        "sensor_value_tolerance": 1e-8,
        "gradient_tolerance": 1e-6,
    },
    "laplace": {"n_samples": 100000, "random_seed": 20260802},
    "validation_reference_file": "",
    "output_path": "output_adjoint_laplace",
    "write_excel": True,
    "make_plots": True,
}


# --------------------------------------------------------------------- settings
def _merge(defaults, given, path="adjoint_laplace_inference"):
    out = copy.deepcopy(defaults)
    for key, value in given.items():
        if key not in defaults:
            raise KeyError(f"unknown setting '{path}.{key}'. Allowed: {sorted(defaults)}")
        if isinstance(defaults[key], dict) and key not in ("std_prior",):
            if not isinstance(value, dict):
                raise TypeError(f"'{path}.{key}' must be an object")
            out[key] = _merge(defaults[key], value, f"{path}.{key}")
        else:
            out[key] = value
    return out


def ReadSettings(parameters):
    given = json.loads(parameters["adjoint_laplace_inference"].WriteJsonString())
    s = _merge(DEFAULTS, given)
    if s["mode"] not in ("u_mean", "gauss_hermite"):
        raise ValueError(f"mode must be 'u_mean' or 'gauss_hermite', got '{s['mode']}'")
    return s


def ReadBaseParameters(parameters):
    """forward_model and parameters blocks come from BayesianParameters.json, so the
    model, the zones and the alpha prior are defined in one place only."""
    path = parameters["base_parameters_file"].GetString()
    with open(path, "r") as f:
        base = Kratos.Parameters(f.read())
    return base["forward_model"].Clone(), [base["parameters"][i].Clone()
                                           for i in range(base["parameters"].size())]


def _json_key(path, dotted):
    with open(path, "r") as f:
        value = json.load(f)
    for part in dotted.split("."):
        value = value[part]
    return value


def ReadData(settings, sensor_names):
    path = settings["measured_data_file"]
    with open(path, newline="") as f:
        rows = {r["name"].strip(): r for r in csv.DictReader(f)}
    missing = [n for n in sensor_names if n not in rows]
    if missing:
        raise KeyError(f"{path} has no row for sensor(s) {missing}")

    def column(name):
        if name not in rows[sensor_names[0]]:
            raise KeyError(f"{path} has no '{name}' column: run collapse_phase1.py to "
                           "create measured_data_collapsed.csv")
        return np.array([float(rows[n][name]) for n in sensor_names])

    data = {"file": path, "sensors": list(sensor_names)}
    if settings["mode"] == "u_mean":
        sigma = float(settings["u_mean"]["sigma_override"])
        if sigma <= 0.0:
            sigma = float(_json_key(settings["noise_model_file"], settings["u_mean"]["sigma_key"]))
        data.update({"d": column("value"), "sigma": sigma})
    else:
        n = column("n_samples")
        data.update({"mean": column("u_hat_mean"), "std": column("u_hat_std"),
                     "n_samples": float(n[0]),
                     "sensor_noise_sigma": float(_json_key(
                         settings["noise_model_file"],
                         settings["gauss_hermite"]["sensor_noise_sigma_key"]))})
    return data


# -------------------------------------------------------------- startup checks
def RunStartupChecks(asm, alpha, d, settings, log):
    """Verifies the adjoint wiring before optimising (no finite differences):
    1) Kratos DisplacementSensor values == sensors.py values (same interpolation)
    2) homogeneity identity sum_i alpha_i du/dalpha_i = -u (sign and scale of S)
    3) Kratos misfit adjoint gradient == gradient rebuilt from per-sensor adjoints
    """
    c = settings["checks"]
    u = asm.Evaluate(alpha)
    u_kratos = asm.KratosSensorValues()
    sensor_diff = float(np.max(np.abs(u_kratos - u)) / np.max(np.abs(u)))
    S = asm.SensorJacobian()
    ratio, status = asm.HomogeneityCheck(u, S, c["homogeneity_tolerance"])

    log(f"check 1: sensors.py vs Kratos sensor values, max rel. diff = {sensor_diff:.2e}")
    if sensor_diff > c["sensor_value_tolerance"]:
        raise RuntimeError("Kratos DisplacementSensor and sensors.py read different values; "
                           "check sensor_data.json locations/directions")
    if status == "sign_flipped":
        log("check 2: homogeneity ratio = -1 -> this Kratos build stores the sensitivity "
            "with the opposite sign; sign corrected and re-checked")
        asm.sign *= -1.0
        S = -S
        ratio, status = asm.HomogeneityCheck(u, S, c["homogeneity_tolerance"])
    if status == "failed":
        raise RuntimeError(f"homogeneity check failed: sum alpha_i du/dalpha_i / (-u) = {ratio} "
                           "(expected 1). The adjoint sensitivities are wrong: check the "
                           "adjoint boundary conditions against the primal ones.")
    log(f"check 2: homogeneity sum_i alpha_i du/dalpha_i / (-u) = "
        f"{np.array2string(ratio, precision=8)} ({status})")

    Jk, gk = asm.MisfitValueAndGradient(d)
    g_from_S = S.T @ (u - np.asarray(d, float))
    scale = max(np.max(np.abs(g_from_S)), 1e-300)
    grad_diff = float(np.max(np.abs(gk - g_from_S)) / scale)
    log(f"check 3: Kratos misfit adjoint gradient vs per-sensor adjoints, rel. diff = {grad_diff:.2e}")
    if grad_diff > c["gradient_tolerance"]:
        Kratos.Logger.PrintWarning(PREFIX, f"check 3 above tolerance ({grad_diff:.2e} > "
                                   f"{c['gradient_tolerance']:.0e}): the misfit adjoint gradient "
                                   "is inaccurate, Gauss-Newton will stall near the optimum")
    return {"alpha": alpha.tolist(), "sensor_value_rel_diff": sensor_diff,
            "homogeneity_ratio": None if ratio is None else ratio.tolist(),
            "homogeneity_status": status, "misfit_gradient_rel_diff": grad_diff,
            "sensitivity_sign": asm.sign}


# ----------------------------------------------------------------------- output
def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def WriteHistory(path, history, names):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration"] + names + ["J", "grad_norm", "newton_decrement",
                                            "rel_step", "step_length"])
        for h in history:
            w.writerow([h["iteration"]] + [f"{v:.16e}" for v in h["theta"]]
                       + [f"{h['J']:.16e}", f"{h['grad_norm']:.6e}", f"{h['decrement']:.6e}",
                          f"{h['step_norm']:.6e}", f"{h['step_length']:.6e}"])


def WriteExcel(path, names, lap, summary_rows, history, log):
    try:
        import pandas as pd
        import openpyxl  # noqa: F401
    except ImportError:
        log("pandas/openpyxl missing - skipping xlsx")
        return
    n = min(len(lap["samples"]), 20000)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(lap["samples"][:n], columns=names).to_excel(
            writer, sheet_name="laplace_samples", index=False)
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="posterior_stats", index=False)
        pd.DataFrame([{"iteration": h["iteration"], "J": h["J"], "grad_norm": h["grad_norm"],
                       **{nm: v for nm, v in zip(names, h["theta"])}} for h in history]).to_excel(
            writer, sheet_name="iterations", index=False)


def MakePlots(out, mode, names, lap, refs, zone_names, validation, log):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib missing - skipping plots")
        return
    m, sd = lap["map"], lap["std"]
    if mode == "u_mean":
        nz = len(m)
        fig, axes = plt.subplots(1, nz, figsize=(5.5 * nz, 4), squeeze=False)
        for i in range(nz):
            ax = axes[0][i]
            x = np.linspace(m[i] - 5 * sd[i], m[i] + 5 * sd[i], 400)
            ax.plot(x * refs[i] / 1e9, np.exp(-0.5 * ((x - m[i]) / sd[i]) ** 2), "b-",
                    label="Laplace posterior")
            ax.set_xlabel(f"E [{zone_names[i]}] [GPa]")
            ax.set_ylabel("relative density")
            ax.set_title(f"E = {m[i] * refs[i] / 1e9:.3f} +/- {sd[i] * refs[i] / 1e9:.3f} GPa")
            if validation and "alpha_mean" in validation:
                ax.axvline(validation["alpha_mean"] * refs[i] / 1e9, color="k", ls="--",
                           label="mean of Phase 1 E (validation only)")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "posterior_alpha_laplace.png"), dpi=150)
        plt.close(fig)
        return

    nz = len(m) // 2
    fig, axes = plt.subplots(1, nz, figsize=(6 * nz, 4), squeeze=False)
    rng = np.random.default_rng(0)
    draws = lap["samples"][rng.choice(len(lap["samples"]), size=min(400, len(lap["samples"])),
                                      replace=False)]
    for i in range(nz):
        ax = axes[0][i]
        mu, s = m[i], m[nz + i]
        x = np.linspace(mu - 4.5 * s, mu + 4.5 * s, 400)
        curves = np.array([np.exp(-0.5 * ((x - dm) / ds) ** 2) / (ds * np.sqrt(2 * np.pi))
                           for dm, ds in zip(draws[:, i], draws[:, nz + i]) if ds > 0])
        E = x * refs[i] / 1e9
        scale = 1e9 / refs[i]
        ax.fill_between(E, np.percentile(curves, 2.5, axis=0) * scale,
                        np.percentile(curves, 97.5, axis=0) * scale, color="b", alpha=0.2,
                        label="95% band (Laplace)")
        ax.plot(E, np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi)) * scale, "b-",
                label=f"recovered N({mu * refs[i] / 1e9:.2f}, {s * refs[i] / 1e9:.2f}) GPa")
        if validation and "alpha_mean" in validation:
            vm, vs = validation["alpha_mean"], validation["alpha_sd"]
            ax.plot(E, np.exp(-0.5 * ((x - vm) / vs) ** 2) / (vs * np.sqrt(2 * np.pi)) * scale,
                    "k--", label="Phase 1 sample E (validation only)")
        ax.set_xlabel(f"E [{zone_names[i]}] [GPa]")
        ax.set_ylabel("density [1/GPa]")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "population_laplace.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, nz, figsize=(5 * nz, 4.5), squeeze=False)
    for i in range(nz):
        ax = axes[0][i]
        ax.scatter(lap["samples"][:3000, i] * refs[i] / 1e9,
                   lap["samples"][:3000, nz + i] * refs[i] / 1e9, s=3, alpha=0.3)
        ax.plot(m[i] * refs[i] / 1e9, m[nz + i] * refs[i] / 1e9, "r*", ms=12, label="MAP")
        ax.set_xlabel(f"population mean E [{zone_names[i]}] [GPa]")
        ax.set_ylabel(f"population sd E [{zone_names[i]}] [GPa]")
        ax.set_title(f"correlation = {lap['correlation'][i, nz + i]:+.3f}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "laplace_parameters.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------ run
def RunAdjointLaplace(parameters):
    from adjoint_sensitivity_model import AdjointSensitivityModel   # Kratos-dependent

    def log(msg):
        Kratos.Logger.PrintInfo(PREFIX, msg)

    settings = ReadSettings(parameters)
    forward_settings, entries = ReadBaseParameters(parameters)
    zone_names = [e["name"].GetString() for e in entries]
    alpha_specs = [json.loads(e["prior"].WriteJsonString()) for e in entries]
    mode = settings["mode"]

    model = Kratos.Model()
    asm = AdjointSensitivityModel(model, forward_settings, entries,
                                  settings["adjoint_parameters_file"], settings["element_name"])
    data = ReadData(settings, asm.sensor_names)
    nz = asm.n_zones

    if mode == "u_mean":
        names = list(zone_names)
        prior = Prior(alpha_specs, names)
        theta0 = np.array(settings["u_mean"]["initial_guess"], float)
        objective = UMeanObjective(asm, data["d"], data["sigma"], prior, log=log)
        check_alpha, check_d = theta0, data["d"]
        log(f"u_mean mode: d = {data['d']}, sigma = {data['sigma']:.6e}")
    else:
        gh = settings["gauss_hermite"]
        names = [f"mu_{z}" for z in zone_names] + [f"sigma_{z}" for z in zone_names]
        prior = Prior(alpha_specs + [gh["std_prior"]] * nz, names)
        theta0 = np.array(list(gh["initial_mean"]) + list(gh["initial_std"]), float)
        objective = GaussHermiteMomentObjective(
            asm, data, prior, nz, order=gh["order"], match=gh["match"],
            min_node_alpha=gh["min_node_alpha"], log=log)
        check_alpha, check_d = np.array(gh["initial_mean"], float), data["mean"]
        log(f"gauss_hermite mode: {len(objective.W)} points per evaluation, matching "
            f"{list(objective.match)}; n = {data['n_samples']:.0f}, "
            f"sensor noise = {data['sensor_noise_sigma']:.3e}")
    if len(theta0) != len(names):
        raise ValueError(f"initial values have {len(theta0)} entries, expected {len(names)}")

    checks = {}
    if settings["checks"]["run_startup_checks"]:
        checks = RunStartupChecks(asm, check_alpha, check_d, settings, log)

    log(f"optimiser: {settings['optimizer']['type']}")
    optimizer = Optimizer(objective.evaluate, prior, settings["optimizer"], log=log)
    ev, history, converged, reason = optimizer.run(theta0)
    log(f"optimiser finished: converged = {converged} ({reason}), {len(history) - 1} iterations")

    lap = laplace_posterior(ev, prior, settings["laplace"]["n_samples"],
                            settings["laplace"]["random_seed"])
    for w in lap["warnings"]:
        Kratos.Logger.PrintWarning(PREFIX, w)

    validation = {}
    if settings["validation_reference_file"]:
        with open(settings["validation_reference_file"], "r") as f:
            ref = json.load(f)
        validation = {k: ref[k] for k in ("alpha_mean", "alpha_sd") if k in ref}
        validation["note"] = ("Phase 1 alpha statistics, read after the run for comparison "
                              "only; never used in the inference")

    refs = asm.refs
    m, sd = lap["map"], lap["std"]
    summary_rows = []
    if mode == "u_mean":
        for i, z in enumerate(zone_names):
            summary_rows.append({
                "zone": i + 1, "name": z, "alpha_mean": m[i], "alpha_std": sd[i],
                "alpha_p2.5": m[i] - 1.959964 * sd[i], "alpha_p97.5": m[i] + 1.959964 * sd[i],
                "E_ref_Pa": refs[i], "E_mean_Pa": m[i] * refs[i], "E_std_Pa": sd[i] * refs[i]})
            log(f"zone {i + 1}: alpha = {m[i]:.6f} +/- {sd[i]:.6f}   "
                f"E = {m[i] * refs[i]:.4e} +/- {sd[i] * refs[i]:.3e} Pa")
        fit = {"u_predicted": ev.info["u"], "u_data": data["d"],
               "residual_over_sigma": ev.info["residual_over_sigma"]}
    else:
        for i, z in enumerate(zone_names):
            row = {"zone": i + 1, "name": z,
                   "mu_alpha": m[i], "mu_alpha_std": sd[i],
                   "sigma_alpha": m[nz + i], "sigma_alpha_std": sd[nz + i],
                   "corr_mu_sigma": lap["correlation"][i, nz + i],
                   "E_ref_Pa": refs[i], "mu_E_Pa": m[i] * refs[i], "mu_E_std_Pa": sd[i] * refs[i],
                   "sigma_E_Pa": m[nz + i] * refs[i], "sigma_E_std_Pa": sd[nz + i] * refs[i]}
            summary_rows.append(row)
            log(f"zone {i + 1}: population mean alpha = {m[i]:.6f} +/- {sd[i]:.6f}, "
                f"population sd alpha = {m[nz + i]:.6f} +/- {sd[nz + i]:.6f}  "
                f"(E: {m[i] * refs[i] / 1e9:.3f} GPa, sd {m[nz + i] * refs[i] / 1e9:.3f} GPa)")
        fit = {"gauss_hermite_alphas": ev.info["alphas"], "gauss_hermite_u": ev.info["U"],
               "gauss_hermite_weights": objective.W,
               "u_mean_predicted": ev.info["pred_mean"], "u_mean_data": data["mean"],
               "u_std_predicted": ev.info["pred_std"], "u_std_data": data["std"],
               "residual": ev.info["residual"]}
    if validation and "alpha_mean" in validation:
        validation["rel_error_mean"] = float(m[0] / validation["alpha_mean"] - 1.0)
        if mode == "gauss_hermite" and "alpha_sd" in validation:
            validation["rel_error_sd"] = float(m[nz] / validation["alpha_sd"] - 1.0)

    out = os.path.join(settings["output_path"], mode)
    os.makedirs(out, exist_ok=True)
    summary = {
        "method": f"adjoint {settings['optimizer']['type']} + Laplace",
        "mode": mode, "parameter_names": names,
        "converged": converged, "stop_reason": reason, "iterations": len(history) - 1,
        "n_forward_solves": asm.n_primal, "n_adjoint_solves": asm.n_adjoint,
        "startup_checks": checks,
        "data": {k: v for k, v in data.items()},
        "zones" if mode == "u_mean" else "population": summary_rows,
        "laplace": {k: lap[k] for k in ("map", "std", "covariance", "correlation",
                                        "hessian_eigenvalues", "condition_number",
                                        "gradient_norm", "newton_decrement",
                                        "fraction_outside_prior", "warnings")},
        "fit": fit, "validation": validation, "settings": settings,
    }
    if mode == "u_mean":
        summary["gradient_cross_check_max"] = objective.max_gradient_mismatch
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(_jsonable(summary), f, indent=2)
    np.savez(os.path.join(out, "laplace_posterior.npz"), names=np.array(names),
             map=lap["map"], covariance=lap["covariance"], samples=lap["samples"],
             hessian=lap["hessian"], E_ref=refs,
             history_theta=np.array([h["theta"] for h in history]),
             history_J=np.array([h["J"] for h in history]))
    WriteHistory(os.path.join(out, "iteration_history.csv"), history, names)
    if settings["write_excel"]:
        WriteExcel(os.path.join(out, "posterior.xlsx"), names, lap, summary_rows, history, log)
    if settings["make_plots"]:
        MakePlots(out, mode, names, lap, refs, zone_names, validation, log)

    log(f"{asm.n_primal} forward solves, {asm.n_adjoint} adjoint solves; results in {out}")
    asm.Finalize()
    return summary
