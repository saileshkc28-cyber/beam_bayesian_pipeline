"""Re-solve the Phase 1 realizations whose stored response is wrong or missing.

Diagnostic only. Imports run_forward from MainKratos_phase1.py unchanged, never writes
into phase1_distribution_runs/, and never touches StructuralMaterials.json.

Each E_true is solved twice, back to back, in this one fresh process. A correct
response satisfies  u = u_ref * E_ref / E  exactly (linear static, u ~ 1/E), where
u_ref = median(alpha * u_true) over the valid Phase 1 rows.

  both solves correct          -> the Phase 1 error was not tied to that E value
  both wrong, same value       -> the error is reproducible for that E value
  wrong and different each time -> state or threading makes the solve unreliable

    python diagnose_bad_solves.py                 # the 17 known bad ids, 34 solves
    python diagnose_bad_solves.py 162 212         # any subset of sample ids

Output: diagnostics/resolve_bad_samples.csv plus a console table.
"""
import csv
import os
import sys
import time

import numpy as np

import MainKratos_phase1 as phase1

WRONG = [162, 212, 221, 271, 444, 585, 586, 622, 634, 654, 659, 690, 772, 843, 982]
FAILED = [59, 396]
REPEATS = 2
REL_TOL = 1e-6          # valid Phase 1 rows agree with u_ref to ~1e-11
AGGREGATE = os.path.join(phase1.OUTPUT_DIRECTORY, "phase1_samples.csv")
OUT_DIR = "diagnostics"
OUT_CSV = os.path.join(OUT_DIR, "resolve_bad_samples.csv")


def load_phase1(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    names = [c[len("u_true_"):] for c in rows[0] if c.startswith("u_true_")]
    return rows, names


def reference_response(rows, names):
    """u at alpha = 1, per sensor, from the valid rows; robust to the bad ones."""
    ok = [r for r in rows if r["status"].strip().strip('"') == "ok"]
    scaled = np.array([[float(r["alpha_true"]) * float(r[f"u_true_{s}"]) for s in names]
                       for r in ok])
    return np.median(scaled, axis=0)


def main():
    ids = [int(a) for a in sys.argv[1:]] or sorted(WRONG + FAILED)

    rows, names = load_phase1(AGGREGATE)
    by_id = {int(r["sample_id"]): r for r in rows}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"sample ids not in {AGGREGATE}: {missing}")

    u_ref = reference_response(rows, names)
    E_ref = phase1.read_reference_young_modulus()
    sensors = phase1.read_sensors(phase1.SENSOR_DATA_PATH)
    if [s["name"] for s in sensors] != names:
        raise SystemExit(f"{phase1.SENSOR_DATA_PATH} sensors {[s['name'] for s in sensors]} "
                         f"do not match the Phase 1 record {names}")

    print(f"E_ref = {E_ref:.6e} Pa   u_ref = {u_ref}   OMP_NUM_THREADS = "
          f"{os.environ.get('OMP_NUM_THREADS', 'unset')}")
    print(f"{len(ids)} samples x {REPEATS} solves = {len(ids) * REPEATS} forward solves\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    records = []
    for sid in ids:
        r = by_id[sid]
        E = float(r["E_true"])
        expected = u_ref * E_ref / E
        stored = np.array([float(r[f"u_true_{s}"]) for s in names])
        for rep in range(1, REPEATS + 1):
            start = time.perf_counter()
            try:
                u = np.asarray(phase1.run_forward(E, vtk_output_path=None), dtype=float)
                status = "ok"
            except Exception as exc:
                u = np.full(len(names), np.nan)
                status = " ".join(f"failed: {type(exc).__name__}: {exc}".split())[:300]
            rel = np.abs(u / expected - 1.0)
            records.append({
                "sample_id": sid,
                "phase1_kind": "failed" if sid in FAILED else "wrong",
                "repeat": rep,
                "E_true": E,
                "alpha_true": E / E_ref,
                "status": status,
                "seconds": time.perf_counter() - start,
                "match": bool(status == "ok" and np.all(rel <= REL_TOL)),
                "max_rel_err": float(np.nanmax(rel)) if status == "ok" else float("nan"),
                **{f"u_{s}": float(v) for s, v in zip(names, u)},
                **{f"expected_{s}": float(v) for s, v in zip(names, expected)},
                **{f"phase1_{s}": float(v) for s, v in zip(names, stored)},
            })

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    # console table, first sensor only (the tip on the frozen single-sensor layout)
    s0 = names[0]
    print(f"\n{'id':>4} {'kind':>6} {'alpha':>8} {'expected':>14} "
          f"{'solve 1':>14} {'ok':>3} {'solve 2':>14} {'ok':>3} {'1==2':>5}")
    for sid in ids:
        a, b = [x for x in records if x["sample_id"] == sid][:2]
        same = (a["status"] == b["status"] == "ok"
                and abs(a[f"u_{s0}"] / b[f"u_{s0}"] - 1.0) <= REL_TOL)
        print(f"{sid:4d} {a['phase1_kind']:>6} {a['alpha_true']:8.5f} "
              f"{a[f'expected_{s0}']:14.6e} {a[f'u_{s0}']:14.6e} {'y' if a['match'] else 'N':>3} "
              f"{b[f'u_{s0}']:14.6e} {'y' if b['match'] else 'N':>3} {'y' if same else 'N':>5}")
        for x in (a, b):
            if x["status"] != "ok":
                print(f"      repeat {x['repeat']}: {x['status'][:120]}")

    n_match = sum(x["match"] for x in records)
    print(f"\n{n_match} of {len(records)} solves match u_ref * E_ref / E "
          f"(rel tol {REL_TOL:g})")
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
