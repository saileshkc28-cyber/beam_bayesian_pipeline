"""Diagnose the Phase 1 population dataset.

    python check_phase1_data.py [csv] [--u-ref -1.894...e-06] [--tol 1e-9]

Reports per-column statistics, cross-checks u against the exact beam law
u * alpha = u_ref, and lists the rows that break it.
"""

import argparse
import csv
import json
import os

import numpy as np

DEFAULT_CSV = "phase1_distribution_runs/phase1_samples.csv"
DEFAULT_U_REF = -1.8940482566888715e-06


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", nargs="?", default=DEFAULT_CSV)
    ap.add_argument("--u-ref", type=float, default=DEFAULT_U_REF)
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="relative tolerance on u * alpha = u_ref")
    ap.add_argument("--worst", type=int, default=10)
    args = ap.parse_args()

    with open(args.csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"{args.csv_path}: {len(rows)} rows, columns = {list(rows[0])}")

    status = {}
    for r in rows:
        status[r.get("status", "").strip('"')] = status.get(r.get("status", "").strip('"'), 0) + 1
    print(f"status counts: {status}")

    def col(name):
        return np.array([float(r[name]) for r in rows])

    true_cols = [c for c in rows[0] if c.startswith("u_true_")]
    hat_cols = [c for c in rows[0] if c.startswith("u_hat_")]

    print(f"\n{'column':<22}{'mean':>16}{'sd':>16}{'sd/|mean| %':>14}")
    named = ["xi", "E_true", "alpha_true"] + true_cols + hat_cols
    data = {}
    for name in named:
        if name not in rows[0]:
            continue
        v = col(name)
        data[name] = v
        print(f"{name:<22}{v.mean():>16.8e}{v.std(ddof=1):>16.8e}"
              f"{100 * v.std(ddof=1) / abs(v.mean()):>14.3f}")

    a = data.get("alpha_true")
    if a is None or not true_cols:
        return

    u = data[true_cols[0]]
    resid = u * a / args.u_ref - 1.0
    bad = np.flatnonzero(np.abs(resid) > args.tol)
    print(f"\nexactness of u * alpha = u_ref  (u_ref = {args.u_ref:.12e})")
    print(f"  max |relative error| : {np.abs(resid).max():.3e}")
    print(f"  rows above tol {args.tol:.0e} : {bad.size} of {len(rows)}")
    if bad.size:
        order = bad[np.argsort(-np.abs(resid[bad]))][: args.worst]
        print(f"  {'sample_id':>10}{'alpha':>14}{'u_true':>18}{'rel err':>14}")
        for i in order:
            print(f"  {rows[i].get('sample_id', i):>10}{a[i]:>14.6f}"
                  f"{u[i]:>18.8e}{resid[i]:>14.3e}")

    keep = np.abs(resid) <= args.tol
    print(f"\nrestricted to the {int(keep.sum())} exact rows:")
    for name, v in (("alpha_true", a), (true_cols[0], u)):
        print(f"  {name:<20}mean {v[keep].mean():.8e}  sd {v[keep].std(ddof=1):.8e}")
    if hat_cols:
        uh = data[hat_cols[0]]
        print(f"  {hat_cols[0]:<20}mean {uh[keep].mean():.8e}  sd {uh[keep].std(ddof=1):.8e}")

    print("\npredicted from the alpha column via u = u_ref / alpha:")
    up = args.u_ref / a[keep]
    print(f"  mean {up.mean():.8e}  sd {up.std(ddof=1):.8e}")
    print(f"  E population: mean {(a[keep] * 206.9).mean():.4f} GPa  "
          f"sd {(a[keep] * 206.9).std(ddof=1):.4f} GPa")

    summary = os.path.join(os.path.dirname(args.csv_path),
                           "phase1_response_summary.json")
    if os.path.exists(summary):
        with open(summary) as f:
            s = json.load(f)
        print(f"\nsummary file reports: n = {s.get('n_samples')}, "
              f"alpha_mean = {s.get('alpha_mean')}, alpha_sd = {s.get('alpha_sd')}")
        print(f"  u_sd = {s.get('u_sd')}, u_hat_sd = {s.get('u_hat_sd')}")


if __name__ == "__main__":
    main()
