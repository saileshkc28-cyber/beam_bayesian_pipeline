"""Collapse the Phase 1 response distribution into one per-sensor summary.

Reads phase1_distribution_runs/phase1_samples.csv and writes, per sensor:

    u_mean      mean of the clean responses   over all valid realizations
    u_std       sd   of the clean responses   over all valid realizations
    u_hat_mean  mean of the noisy responses
    u_hat_std   sd   of the noisy responses
    n_samples   how many realizations went in

Output layout deliberately matches measured_data.csv -- same header prefix, same
row order, and a "value" column -- so the existing Likelihood can read it unchanged
(csv.DictReader ignores the extra trailing columns). "value" holds u_hat_mean.

    python collapse_phase1.py
    python collapse_phase1.py phase1_distribution_runs

READ BEFORE USING THIS AS PHASE 2 INPUT
---------------------------------------
Each u_i came from a different E_i, so u_mean is not the response of any single
structure. Two consequences, both quantified in the console output:

  * u = u_ref/alpha is nonlinear, so alpha recovered from u_mean is biased away
    from the mean of the true alphas (the printed "expected bias" line);
  * u_std is BETWEEN-realization spread. The likelihood's sigma is WITHIN-reading
    sensor noise. They are different quantities and the likelihood has no slot for
    the first, so feeding this file recovers a single alpha and no distribution.

The file is written regardless -- it is a useful summary and a fair diagnostic.
Just do not expect the population sd to come back out of an inversion that used it.
"""
import csv
import json
import os
import sys
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "phase1_distribution_runs"

# ------------------------------------------------------------------ read the record
aggregate = os.path.join(root, "phase1_samples.csv")
if not os.path.isfile(aggregate):
    raise SystemExit(f"no Phase 1 record found: {aggregate}")

with open(aggregate, newline="") as f:
    rows = [r for r in csv.DictReader(f) if r.get("status", "").strip().strip('"') == "ok"]
if len(rows) < 2:
    raise SystemExit(f"only {len(rows)} valid samples in {aggregate}")

sensor_names = [c[len("u_true_"):] for c in rows[0] if c.startswith("u_true_")]
u_true = np.array([[float(r[f"u_true_{s}"]) for s in sensor_names] for r in rows])
u_hat = np.array([[float(r[f"u_hat_{s}"]) for s in sensor_names] for r in rows])
alpha = np.array([float(r["alpha_true"]) for r in rows])
n = len(rows)

# sensor type / location columns, taken from any one sample so the layout matches
template = {}
for r in rows:
    path = os.path.join(root, f"sample_{int(r['sample_id']):04d}", "measured_data.csv")
    if os.path.isfile(path):
        with open(path, newline="") as f:
            for line in csv.DictReader(f):
                template[line["name"]] = line
        break

sigma = None
for r in rows:
    path = os.path.join(root, f"sample_{int(r['sample_id']):04d}", "noise_model.json")
    if os.path.isfile(path):
        sigma = float(json.load(open(path))["sigma"])
        break

# ------------------------------------------------------------------ write the file
out_csv = os.path.join(root, "measured_data_collapsed.csv")
with open(out_csv, "w", newline="") as f:
    f.write("#,type,name,location_0,location_1,location_2,value,"
            "u_mean,u_std,u_hat_mean,u_hat_std,n_samples\n")
    for i, name in enumerate(sensor_names, 1):
        meta = template.get(name, {})
        f.write(f"{i},{meta.get('type', 'displacement_sensor')},{name},"
                f"{meta.get('location_0', 0.0)},{meta.get('location_1', 0.0)},"
                f"{meta.get('location_2', 0.0)},"
                f"{u_hat[:, i - 1].mean():.16e},"
                f"{u_true[:, i - 1].mean():.16e},{u_true[:, i - 1].std(ddof=1):.16e},"
                f"{u_hat[:, i - 1].mean():.16e},{u_hat[:, i - 1].std(ddof=1):.16e},"
                f"{n}\n")

# ---------------------------------------------- the canonical whole-run u statistics
# ONE mean and ONE sd of u for the entire run, computed here and nowhere else.
# Every other script reads this file rather than recomputing, so the number quoted in
# a figure, a table and the collapsed CSV can never drift apart.
summary_path = os.path.join(root, "phase1_response_summary.json")
json.dump({
    "n_samples": n,
    "sensors": sensor_names,
    "sensor_noise_sigma": sigma,
    "u_mean": u_true.mean(axis=0).tolist(),
    "u_sd": u_true.std(axis=0, ddof=1).tolist(),
    "u_hat_mean": u_hat.mean(axis=0).tolist(),
    "u_hat_sd": u_hat.std(axis=0, ddof=1).tolist(),
    "alpha_mean": float(alpha.mean()),
    "alpha_sd": float(alpha.std(ddof=1)),
}, open(summary_path, "w"), indent=2)

# a matching noise model, so Phase 2 has a sigma to read if this file is used
out_json = os.path.join(root, "noise_model_collapsed.json")
sigma_mean = float(u_hat[:, 0].std(ddof=1) / np.sqrt(n))
json.dump({
    "sigma": sigma_mean,
    "sigma_options": {
        "sensor_noise": sigma,
        "standard_error_of_the_mean": sigma_mean,
        "between_realization_sd": float(u_hat[:, 0].std(ddof=1)),
    },
    "n_samples": n,
    "u_mean": u_true.mean(axis=0).tolist(),
    "u_std": u_true.std(axis=0, ddof=1).tolist(),
    "u_hat_mean": u_hat.mean(axis=0).tolist(),
    "u_hat_std": u_hat.std(axis=0, ddof=1).tolist(),
}, open(out_json, "w"), indent=2)

print(f"wrote {out_csv}")
print(f"wrote {summary_path}   <- canonical whole-run u mean and sd")
print(f"wrote {out_json}\n")

# ------------------------------------------------------------------ console record
print(f"{'sensor':>16} {'u_mean [um]':>14} {'u_std [um]':>14} "
      f"{'u_hat_mean':>14} {'u_hat_std':>14}")
for i, name in enumerate(sensor_names):
    print(f"{name:>16} {u_true[:, i].mean() * 1e6:>14.6f} "
          f"{u_true[:, i].std(ddof=1) * 1e6:>14.6f} "
          f"{u_hat[:, i].mean() * 1e6:>14.6f} {u_hat[:, i].std(ddof=1) * 1e6:>14.6f}")

print(f"\nrealizations collapsed : {n}")
if sigma:
    print(f"sensor noise sigma     : {sigma:.6e}")
    print(f"between-realization sd : {u_hat[:, 0].std(ddof=1):.6e} "
          f"({u_hat[:, 0].std(ddof=1) / sigma:.1f}x the sensor noise)")

# what an inversion on this file would actually return, predicted in closed form
u_ref = float(np.median(alpha * u_true[:, 0]))
alpha_from_mean = u_ref / u_true[:, 0].mean()
print(f"\nif this file is inverted:")
print(f"  alpha recovered from u_mean : {alpha_from_mean:.6f}")
print(f"  mean of the true alphas     : {alpha.mean():.6f}")
print(f"  expected bias               : {100 * (alpha_from_mean / alpha.mean() - 1):+.2f}% "
      f"(from the 1/alpha nonlinearity)")
print(f"  population sd recoverable   : no -- u_std has no counterpart in the likelihood")
