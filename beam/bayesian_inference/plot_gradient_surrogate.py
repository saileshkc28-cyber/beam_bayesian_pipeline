r"""Plot the gradient-enhanced response surrogate: fitted curve, adjoint training
points with their adjoint slopes, and validation points; print the residuals
against the accuracy gate.

    python plot_gradient_surrogate.py --source <case>\N01 --case <case>\N01_adjoint

Reads only, from --source: SurrogateParameters.json, the validation CSV and
surrogate\gradient_training_responses.csv. Reads the model from
<case>\surrogate\response_surrogate.joblib (build_gradient_surrogate.py).
Writes <case>\surrogate\surrogate_fit.png, never into --source. Runs nothing.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from response_surrogate import ResponseSurrogate
from plot_surrogate import read_points
# gradient_surrogate_study imports BuildSurrogate, which imports Kratos only
# inside build_forward_model, so this is safe
from gradient_surrogate_study import ADJOINT_CSV, load_adjoint
from build_gradient_surrogate import inside, resolve_in

TANGENT_HALF_WIDTH = 0.12   # half length of each slope segment, in ln E


def same_path(a, b):
    return os.path.normcase(str(Path(a).resolve())) == os.path.normcase(str(Path(b).resolve()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path, help="reference case folder")
    ap.add_argument("--case", required=True, type=Path, help="gradient case folder")
    ap.add_argument("--sensor", type=int, default=0)
    args = ap.parse_args()

    src, case = args.source.resolve(), args.case.resolve()
    png = case / "surrogate" / "surrogate_fit.png"
    if inside(png, src):
        sys.exit(f"refusing to write {png}: it lies inside --source {src}")

    with open(src / "SurrogateParameters.json") as f:
        cfg = json.load(f)
    out, d = cfg["output"], cfg["domain"]

    gp = ResponseSurrogate.load(str(case / "surrogate" / "response_surrogate.joblib"))

    # plotting domain from the model, as plot_surrogate.py; must match the source
    if (gp.e_min, gp.e_max) != (float(d["e_min_Pa"]), float(d["e_max_Pa"])):
        sys.exit(f"model domain [{gp.e_min:.6e}, {gp.e_max:.6e}] Pa differs from "
                 f"{src / 'SurrogateParameters.json'} "
                 f"[{d['e_min_Pa']:.6e}, {d['e_max_Pa']:.6e}] Pa")

    # the model must have been fitted on the adjoint points plotted here
    grad_csv = src / "surrogate" / ADJOINT_CSV
    fitted_on = (getattr(gp, "identity_", None) or {}).get("training_csv")
    if not fitted_on or not same_path(fitted_on, grad_csv):
        sys.exit(f"model was fitted on {fitted_on}, not on {grad_csv}")

    e_va, u_va = read_points(resolve_in(src, out["validation_csv"]))
    n_sensors = u_va.shape[1]
    e_tr, u_tr, dudt_tr = load_adjoint(grad_csv, n_sensors)   # raises on status != ok

    sigma = np.asarray(cfg["accuracy_gate"]["sigma_noise_m"], float).ravel()
    gate = cfg["accuracy_gate"]["max_error_over_sigma"] * sigma[args.sensor]

    e_fine = np.exp(np.linspace(np.log(gp.e_min), np.log(gp.e_max), 600))
    mean = gp.predict(e_fine)   # no predictive std: no 95% band
    k = args.sensor

    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    ax.plot(e_fine / 1e9, mean[:, k] * 1e6, "-", color="C0", lw=1.6,
            label="gradient-enhanced GP surrogate")
    ax.plot(e_tr / 1e9, u_tr[:, k] * 1e6, "o", ms=5, color="C3",
            label=f"training solves (n={e_tr.size})")
    # adjoint slope through each training point: u_i + dudt_i * (ln E - ln E_i)
    for i, (e_i, u_i, g_i) in enumerate(zip(e_tr, u_tr[:, k], dudt_tr[:, k])):
        dt = np.linspace(-TANGENT_HALF_WIDTH, TANGENT_HALF_WIDTH, 2)
        ax.plot(e_i * np.exp(dt) / 1e9, (u_i + g_i * dt) * 1e6, "-", color="C3",
                lw=1.1, label="adjoint slope" if i == 0 else "_nolegend_")
    ax.plot(e_va / 1e9, u_va[:, k] * 1e6, "x", ms=7, mew=1.6, color="C2",
            label=f"validation solves (n={e_va.size})")
    ax.set_xscale("log")
    ax.set_xlabel("Young's modulus  $E$  [GPa]")
    ax.set_ylabel(r"tip deflection  $u$  [$\mu$m]")
    # training solves behind the fit: one primal per adjoint point, one adjoint
    # per point and sensor (adjoint_training_solves.py)
    ax.set_title(f"Gradient-enhanced response surrogate  $u(E)$:  {e_tr.size} forward"
                 f" + {e_tr.size * n_sensors} adjoint solves")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)

    res_tr = gp.predict(e_tr)[:, k] - u_tr[:, k]
    res_va = gp.predict(e_va)[:, k] - u_va[:, k]

    fig.tight_layout()
    fig.savefig(png, dpi=160)
    print(f"max |training residual|   {np.abs(res_tr).max():.3e} m")
    print(f"max |validation residual| {np.abs(res_va).max():.3e} m  "
          f"gate {gate:.3e} m  -> {'PASS' if np.abs(res_va).max() < gate else 'FAIL'}")
    print(f"-> {png}")


if __name__ == "__main__":
    main()
