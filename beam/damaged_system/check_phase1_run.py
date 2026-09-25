"""Sanity checks on a finished Phase 1 run. No Kratos; run after MainKratos_phase1.py
and clean_phase1.py.

    python check_phase1_run.py <folder> --sensors ../sensor_placement/sensor_data_10s.json

Prints
  * whether xi and E_true match the reference run (default phase1_distribution_runs)
  * sigma per specimen and sensor, plus the realized noise (u_hat - u_true) / sigma
  * whether the u_hat_* column order equals the sensor file order
  * the number of failed and valid specimens
  * the worst alpha * u deviation per sensor
"""
import argparse
import csv
import json
import os

import numpy as np

EXPECTED_SIGMA = 3.788e-08


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description="Phase 1 run checks")
    ap.add_argument("root", help="Phase 1 output folder")
    ap.add_argument("--sensors", required=True, help="sensor layout JSON used for the run")
    ap.add_argument("--reference", default="phase1_distribution_runs",
                    help="earlier run to compare xi and E_true with (default: %(default)s)")
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    rows = read_rows(os.path.join(args.root, "phase1_samples.csv"))
    header = list(rows[0])
    with open(args.sensors) as f:
        sensor_names = [s["name"] for s in json.load(f)["list_of_sensors"]]
    true_cols = [c for c in header if c.startswith("u_true_")]
    hat_cols = [c for c in header if c.startswith("u_hat_")]
    names = [c[len("u_hat_"):] for c in hat_cols]
    ok = [r["status"].strip().strip('"') == "ok" for r in rows]

    # ------------------------------------------------ 1. same xi and E as the reference
    print(f"== 1. xi and E_true vs {args.reference}")
    ref_path = os.path.join(args.reference, "phase1_samples.csv")
    if os.path.isfile(ref_path):
        ref = {int(r["sample_id"]): r for r in read_rows(ref_path)}
        common = [r for r in rows if int(r["sample_id"]) in ref]
        for col in ("xi", "E_true"):
            diff = [abs(float(r[col]) - float(ref[int(r["sample_id"])][col])) for r in common]
            n_bad = sum(d != 0.0 for d in diff)
            print(f"   {col:<7} {len(common) - n_bad} of {len(common)} identical, "
                  f"max |diff| {max(diff):.3e}   -> {'MATCH' if n_bad == 0 else 'MISMATCH'}")
        if len(common) != len(rows) or len(common) != len(ref):
            print(f"   rows: this run {len(rows)}, reference {len(ref)}, common {len(common)}")
    else:
        print(f"   reference not found: {ref_path}")

    # ------------------------------------------------ 2. sigma per specimen and sensor
    print(f"\n== 2. sigma (expected {EXPECTED_SIGMA:.4e} for every specimen and sensor)")
    sigmas, missing = {}, 0
    for r in rows:
        path = os.path.join(args.root, f"sample_{int(r['sample_id']):04d}", "noise_model.json")
        if not os.path.isfile(path):
            missing += 1
            continue
        with open(path) as f:
            s = json.load(f)["sigma"]
        # one scalar sigma per specimen is applied to all sensors; a list would be per sensor
        for v in np.atleast_1d(s):
            sigmas[float(v)] = sigmas.get(float(v), 0) + 1
    print(f"   distinct sigma values: " +
          ", ".join(f"{k:.6e} (x{n})" for k, n in sorted(sigmas.items())) +
          f"   folders without noise_model.json: {missing}")
    print(f"   -> {'OK' if set(sigmas) == {EXPECTED_SIGMA} else 'CHECK'}")
    good = [r for r, k in zip(rows, ok) if k]
    sigma = next(iter(sigmas)) if len(sigmas) == 1 else EXPECTED_SIGMA
    print(f"   realized noise z = (u_hat - u_true) / sigma per sensor, n = {len(good)}:")
    for s in names:
        z = np.array([(float(r[f"u_hat_{s}"]) - float(r[f"u_true_{s}"])) / sigma
                      for r in good])
        print(f"     {s:<16} mean {z.mean():+.3f}   sd {z.std(ddof=1):.3f}")

    # ------------------------------------------------ 3. column order
    print("\n== 3. column order")
    same_hat = names == sensor_names
    same_true = [c[len("u_true_"):] for c in true_cols] == sensor_names
    print(f"   sensor file : {sensor_names}")
    print(f"   u_hat_*     : {names}")
    print(f"   u_hat order == sensor file: {same_hat}   u_true order == sensor file: "
          f"{same_true}")
    probe = next((r for r, k in zip(rows, ok) if k), None)
    if probe is not None:
        path = os.path.join(args.root, f"sample_{int(probe['sample_id']):04d}",
                            "measured_data.csv")
        if os.path.isfile(path):
            file_names = [r["name"] for r in read_rows(path)]
            print(f"   sample_{int(probe['sample_id']):04d}/measured_data.csv row order == "
                  f"sensor file: {file_names == sensor_names}")

    # ------------------------------------------------ 4. failed and valid
    print("\n== 4. failed and valid specimens")
    failed = [int(r["sample_id"]) for r, k in zip(rows, ok) if not k]
    print(f"   rows {len(rows)}   solver failed {len(failed)} {failed}")
    clean_path = os.path.join(args.root, "phase1_samples_clean.csv")
    if os.path.isfile(clean_path):
        clean = read_rows(clean_path)
        invalid = [int(r["sample_id"]) for r in clean if r["valid"] != "1"]
        print(f"   valid (phase1_samples_clean.csv) {len(clean) - len(invalid)}   "
              f"invalid {len(invalid)} {invalid}")
    else:
        print(f"   {clean_path} not found -- run clean_phase1.py first")

    # ------------------------------------------------ 5. worst alpha*u deviation
    print(f"\n== 5. worst |alpha * u_true / median - 1| per sensor over solver-ok rows "
          f"(tol {args.tol:g})")
    alpha = np.array([float(r["alpha_true"]) for r in good])
    for s in names:
        scaled = alpha * np.array([float(r[f"u_true_{s}"]) for r in good])
        dev = np.abs(scaled / np.median(scaled) - 1.0)
        worst = int(np.argmax(dev))
        print(f"   {s:<16} worst {dev[worst]:.3e} (sample {good[worst]['sample_id']})   "
              f"above tol: {int((dev >= args.tol).sum())}")


if __name__ == "__main__":
    main()
