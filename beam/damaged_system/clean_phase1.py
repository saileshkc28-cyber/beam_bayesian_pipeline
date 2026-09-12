"""Phase 1 quality control.

    python clean_phase1.py [csv] [--u-ref ...] [--tol 1e-9]

Flags every row that fails the exact beam law u * alpha = u_ref, writes
phase1_samples_clean.csv with a valid column, and records the realized
population truth in phase1_clean_summary.json.

This step belongs to Phase 1 because the test uses alpha_true. Phase 2 must
read only the u_hat column of the valid rows.
"""

import argparse
import csv
import json
import os

import numpy as np

DEFAULT_CSV = "phase1_distribution_runs/phase1_samples.csv"
DEFAULT_U_REF = -1.8940482566888715e-06
E_REF_PA = 206.9e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", nargs="?", default=DEFAULT_CSV)
    ap.add_argument("--u-ref", type=float, default=DEFAULT_U_REF)
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    with open(args.csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0])
    true_col = next(c for c in fields if c.startswith("u_true_"))
    hat_col = next(c for c in fields if c.startswith("u_hat_"))

    def as_float(r, name):
        try:
            return float(r[name])
        except (TypeError, ValueError):
            return np.nan

    alpha = np.array([as_float(r, "alpha_true") for r in rows])
    u = np.array([as_float(r, true_col) for r in rows])
    uh = np.array([as_float(r, hat_col) for r in rows])

    solver_ok = np.array([r.get("status", "").strip('"') == "ok" for r in rows])
    finite = np.isfinite(u) & np.isfinite(uh) & np.isfinite(alpha)
    with np.errstate(invalid="ignore"):
        resid = u * alpha / args.u_ref - 1.0
    exact = finite & (np.abs(resid) <= args.tol)
    valid = solver_ok & exact

    out_csv = args.csv_path.replace(".csv", "_clean.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields + ["valid", "law_residual"])
        w.writeheader()
        for i, r in enumerate(rows):
            row = dict(r)
            row["valid"] = int(valid[i])
            row["law_residual"] = repr(float(resid[i]))
            w.writerow(row)

    excluded = [
        {"sample_id": rows[i].get("sample_id"),
         "reason": "solver_failed" if not solver_ok[i]
                   else ("non_finite" if not finite[i] else "law_violation"),
         "alpha_true": None if not np.isfinite(alpha[i]) else float(alpha[i]),
         "u_true": None if not np.isfinite(u[i]) else float(u[i]),
         "law_residual": None if not np.isfinite(resid[i]) else float(resid[i])}
        for i in np.flatnonzero(~valid)
    ]

    a, uu, uhh = alpha[valid], u[valid], uh[valid]
    e = a * E_REF_PA
    summary = {
        "source_file": os.path.basename(args.csv_path),
        "n_rows": len(rows),
        "n_valid": int(valid.sum()),
        "n_excluded": int((~valid).sum()),
        "exclusion_rule": {
            "status": "ok",
            "law": "abs(u_true * alpha_true / u_ref - 1) <= tol",
            "u_ref_m": args.u_ref,
            "tol": args.tol,
        },
        "sensor": true_col.replace("u_true_", ""),
        "sensor_noise_sigma": 3.788e-08,
        "population_truth": {
            "E_mean_Pa": float(e.mean()), "E_sd_Pa": float(e.std(ddof=1)),
            "E_mean_GPa": float(e.mean() / 1e9), "E_sd_GPa": float(e.std(ddof=1) / 1e9),
            "alpha_mean": float(a.mean()), "alpha_sd": float(a.std(ddof=1)),
            "e_ref_Pa": E_REF_PA,
        },
        "response": {
            "u_mean": float(uu.mean()), "u_sd": float(uu.std(ddof=1)),
            "u_hat_mean": float(uhh.mean()), "u_hat_sd": float(uhh.std(ddof=1)),
        },
        "excluded": excluded,
    }
    out_json = os.path.join(os.path.dirname(args.csv_path), "phase1_clean_summary.json")
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
    pred = np.hypot(uu.std(ddof=1), summary["sensor_noise_sigma"])
    print(f"\nnoise consistency  sqrt(u_sd^2 + sigma^2) = {pred:.6e}")
    print(f"                   u_hat_sd                = {uhh.std(ddof=1):.6e}"
          f"   ({100 * (uhh.std(ddof=1) / pred - 1):+.2f}%)")
    print(f"\n-> {out_csv}\n-> {out_json}")


if __name__ == "__main__":
    main()
