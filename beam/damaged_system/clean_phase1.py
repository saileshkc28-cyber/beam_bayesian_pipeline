"""Phase 1 quality control.

    python clean_phase1.py [folder] [--tol 1e-6]

Flags every row that fails the exact beam law alpha * u_true = u_ref, checked for
EVERY sensor, writes <folder>/phase1_samples_clean.csv with a valid column, and
records the realized population truth in <folder>/phase1_clean_summary.json.

u_ref is the per-sensor median of alpha_true * u_true over the rows the solver
reported as ok, so no reference value is hard-coded and any sensor layout works.
A row is valid only if status == ok, every value is finite, and
|alpha * u_true / u_ref - 1| < tol for every sensor.

Output format follows clean_phase1.py on hierarchical_phase_2: the input columns plus
"valid" and "law_residual" (the worst sensor's signed residual). With more than one
sensor, law_residual_<name> columns are appended per sensor.

This step belongs to Phase 1 because the test uses alpha_true. Phase 2 must read only
the u_hat columns of the valid rows.
"""
import argparse
import csv
import json
import os

import numpy as np

DEFAULT_ROOT = "phase1_distribution_runs"
DEFAULT_TOL = 1e-6


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def read_sigma(root, sample_ids):
    """Sensor noise sigma from the first sample folder that records it."""
    for sid in sample_ids:
        path = os.path.join(root, f"sample_{int(sid):04d}", "noise_model.json")
        if os.path.isfile(path):
            with open(path) as f:
                return float(json.load(f)["sigma"])
    return None


def main():
    ap = argparse.ArgumentParser(description="Phase 1 quality control")
    ap.add_argument("root", nargs="?", default=DEFAULT_ROOT,
                    help="Phase 1 output folder (default: %(default)s)")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL)
    args = ap.parse_args()

    csv_path = os.path.join(args.root, "phase1_samples.csv")
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0])
    names = [c[len("u_true_"):] for c in fields if c.startswith("u_true_")]

    alpha = np.array([as_float(r["alpha_true"]) for r in rows])
    E = np.array([as_float(r["E_true"]) for r in rows])
    u = np.array([[as_float(r[f"u_true_{s}"]) for s in names] for r in rows])
    uh = np.array([[as_float(r[f"u_hat_{s}"]) for s in names] for r in rows])

    solver_ok = np.array([r.get("status", "").strip().strip('"') == "ok" for r in rows])
    finite = np.isfinite(alpha) & np.all(np.isfinite(u), axis=1) & \
        np.all(np.isfinite(uh), axis=1)

    base = solver_ok & finite
    u_ref = np.median(alpha[base, None] * u[base], axis=0)
    with np.errstate(invalid="ignore"):
        resid = alpha[:, None] * u / u_ref - 1.0          # rows x sensors
    exact = finite & np.all(np.abs(resid) < args.tol, axis=1)
    valid = solver_ok & exact

    worst_col = np.nanargmax(np.where(np.isfinite(resid), np.abs(resid), -1.0), axis=1)
    worst = resid[np.arange(len(rows)), worst_col]

    out_csv = os.path.join(args.root, "phase1_samples_clean.csv")
    extra = [f"law_residual_{s}" for s in names] if len(names) > 1 else []
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields + ["valid", "law_residual"] + extra)
        w.writeheader()
        for i, r in enumerate(rows):
            row = dict(r)
            row["valid"] = int(valid[i])
            row["law_residual"] = repr(float(worst[i]))
            for j, s in enumerate(extra):
                row[s] = repr(float(resid[i, j]))
            w.writerow(row)

    def reason(i):
        if not solver_ok[i]:
            return "solver_failed"
        if not finite[i]:
            return "non_finite"
        return "law_violation"

    excluded = [
        {"sample_id": rows[i].get("sample_id"),
         "reason": reason(i),
         "alpha_true": None if not np.isfinite(alpha[i]) else float(alpha[i]),
         "u_true": [None if not np.isfinite(v) else float(v) for v in u[i]],
         "law_residual": None if not np.isfinite(worst[i]) else float(worst[i]),
         "worst_sensor": names[int(worst_col[i])] if np.isfinite(worst[i]) else None}
        for i in np.flatnonzero(~valid)
    ]

    sigma = read_sigma(args.root, [r["sample_id"] for r in rows])
    a, e, uu, uhh = alpha[valid], E[valid], u[valid], uh[valid]
    per_sensor = {
        s: {"u_ref_m": float(u_ref[j]),
            "u_mean": float(uu[:, j].mean()), "u_sd": float(uu[:, j].std(ddof=1)),
            "u_hat_mean": float(uhh[:, j].mean()), "u_hat_sd": float(uhh[:, j].std(ddof=1)),
            "worst_abs_law_residual_valid": float(np.abs(resid[valid, j]).max())}
        for j, s in enumerate(names)}
    summary = {
        "source_file": os.path.basename(csv_path),
        "n_rows": len(rows),
        "n_valid": int(valid.sum()),
        "n_excluded": int((~valid).sum()),
        "exclusion_rule": {
            "status": "ok",
            "law": "abs(alpha_true * u_true / u_ref - 1) < tol for every sensor",
            "u_ref": "per-sensor median of alpha_true * u_true over status-ok rows",
            "u_ref_m": [float(v) for v in u_ref],
            "tol": args.tol,
        },
        "sensor": names[0],
        "sensors": names,
        "sensor_noise_sigma": sigma,
        "population_truth": {
            "E_mean_Pa": float(e.mean()), "E_sd_Pa": float(e.std(ddof=1)),
            "E_mean_GPa": float(e.mean() / 1e9), "E_sd_GPa": float(e.std(ddof=1) / 1e9),
            "alpha_mean": float(a.mean()), "alpha_sd": float(a.std(ddof=1)),
            "e_ref_Pa": float(np.median(e / a)),
        },
        "response": {k: v for k, v in per_sensor[names[0]].items()
                     if k in ("u_mean", "u_sd", "u_hat_mean", "u_hat_sd")},
        "response_per_sensor": per_sensor,
        "excluded": excluded,
    }
    out_json = os.path.join(args.root, "phase1_clean_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    reasons = {}
    for x in excluded:
        reasons[x["reason"]] = reasons.get(x["reason"], 0) + 1
    print(f"{len(rows)} rows -> {int(valid.sum())} valid, "
          f"{int((~valid).sum())} excluded {reasons}")
    print(f"excluded sample_ids: "
          f"{sorted(int(x['sample_id']) for x in excluded if x['sample_id'])}")
    print(f"\npopulation truth (valid rows)")
    print(f"  E      mean {e.mean() / 1e9:10.4f} GPa   sd {e.std(ddof=1) / 1e9:9.4f} GPa")
    print(f"  alpha  mean {a.mean():10.6f}       sd {a.std(ddof=1):9.6f}")
    if sigma:
        print(f"\nnoise consistency, sigma = {sigma:.4e}: u_hat_sd vs sqrt(u_sd^2 + sigma^2)")
        for j, s in enumerate(names):
            pred = np.hypot(uu[:, j].std(ddof=1), sigma)
            got = uhh[:, j].std(ddof=1)
            print(f"  {s:<16} {got:.6e}  vs  {pred:.6e}   ({100 * (got / pred - 1):+.2f}%)")
    print(f"\n-> {out_csv}\n-> {out_json}")


if __name__ == "__main__":
    main()
