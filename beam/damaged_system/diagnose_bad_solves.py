"""Re-solve the Phase 1 realizations whose stored response is wrong or missing.

Diagnostic only. Imports run_forward from MainKratos_phase1.py unchanged, never writes
into phase1_distribution_runs/, and never touches StructuralMaterials.json.

A correct response satisfies  u = u_ref * E_ref / E  exactly (linear static, u ~ 1/E),
where u_ref = median(alpha * u_true) over the valid Phase 1 rows.

  both solves correct          -> the Phase 1 error was not tied to that E value
  both wrong, same value       -> the error is reproducible for that E value
  wrong and different each time -> state or threading makes the solve unreliable

    python diagnose_bad_solves.py                 # the 17 known bad ids, 34 solves
    python diagnose_bad_solves.py 162 212         # any subset of sample ids

Variant testing (a modified copy of PrimalParametersBeam.json):

    python diagnose_bad_solves.py --params diagnostics/PrimalParametersBeam_varA.json \
        --repeats 1 --good 20 --out diagnostics/variant_A.csv

  --good N adds N valid Phase 1 samples drawn with a fixed seed; those are checked
  against their stored u_true (GOOD_TOL) instead of against u_ref.

Equation map, no solve (one Initialize plus one assembly of K):

    python diagnose_bad_solves.py --map-dofs [--params FILE] [--map-sample 59]

Output: a CSV (default diagnostics/resolve_bad_samples.csv) plus a console table.
"""
import argparse
import csv
import os
import random
import time

import numpy as np

import MainKratos_phase1 as phase1

WRONG = [162, 212, 221, 271, 444, 585, 586, 622, 634, 654, 659, 690, 772, 843, 982]
FAILED = [59, 396]
REPEATS = 2
REL_TOL = 1e-6          # valid Phase 1 rows agree with u_ref to ~1e-11
GOOD_TOL = 1e-10        # a good sample must reproduce its stored u_true this closely
GOOD_SEED = 20260802
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


def good_ids(rows, names, u_ref, n, seed):
    """n valid samples whose stored response matches u_ref / alpha, drawn reproducibly."""
    pool = []
    for r in rows:
        if r["status"].strip().strip('"') != "ok" or int(r["sample_id"]) in WRONG + FAILED:
            continue
        u = np.array([float(r[f"u_true_{s}"]) for s in names])
        if np.all(np.abs(float(r["alpha_true"]) * u / u_ref - 1.0) <= REL_TOL):
            pool.append(int(r["sample_id"]))
    return sorted(random.Random(seed).sample(sorted(pool), n))


# ------------------------------------------------------------------- equation map
def _get(obj, name):
    """Kratos binds some Dof members as properties and some as methods."""
    value = getattr(obj, name)
    return value() if callable(value) else value


def map_dofs(E_value, out_csv):
    """Assemble K at E_value (Dirichlet applied, no solve) and report the equation map,
    the smallest singular values and the DOFs carrying the near-null vector."""
    import KratosMultiphysics as Kratos
    from KratosMultiphysics import scipy_conversion_tools

    model = Kratos.Model()
    with open(phase1.PARAMETER_FILE) as f:
        parameters = Kratos.Parameters(f.read())
    for name in ("sensor_output", "vtk_output"):
        if parameters["output_processes"].Has(name):
            parameters["output_processes"].RemoveValue(name)

    analysis = phase1.CustomStructuralMechanicsAnalysis(model, parameters)
    analysis.Initialize()
    structure = model["Structure"]
    for element in structure.Elements:
        if element.Properties.Id == phase1.PROPERTY_ID:
            element.Properties.SetValue(Kratos.YOUNG_MODULUS, float(E_value))

    solver = analysis._GetSolver()
    analysis.time = solver.AdvanceInTime(analysis.time)
    analysis.InitializeSolutionStep()          # BCs applied, DOF set and system set up

    strategy = solver._GetSolutionStrategy()
    builder = solver._GetBuilderAndSolver()
    scheme = solver._GetScheme()
    mp = solver.GetComputingModelPart()
    A = strategy.GetSystemMatrix()
    b = strategy.GetSystemVector()
    dx = strategy.GetSolutionVector()
    builder.Build(scheme, mp, A, b)
    builder.ApplyDirichletConditions(scheme, mp, A, dx, b)

    K = scipy_conversion_tools.to_csr(A).toarray()
    n = K.shape[0]
    dofs = []
    for dof in builder.GetDofSet():
        node = mp.GetNode(_get(dof, "Id"))
        dofs.append({"equation_id": int(_get(dof, "EquationId")),
                     "node": int(_get(dof, "Id")),
                     "x": node.X0, "y": node.Y0, "z": node.Z0,
                     "variable": dof.GetVariable().Name(),
                     "fixed": bool(dof.IsFixed())})
    dofs.sort(key=lambda d: d["equation_id"])
    for d in dofs:
        d["K_diag"] = float(K[d["equation_id"], d["equation_id"]])

    s = np.linalg.svd(K, compute_uv=False)
    _, _, vt = np.linalg.svd(K)
    null = vt[-1]
    cond = s[0] / s[-1] if s[-1] > 0 else float("inf")
    for d in dofs:
        d["null_vector"] = float(null[d["equation_id"]])

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dofs[0]))
        writer.writeheader()
        writer.writerows(dofs)

    print(f"\nparameter file : {phase1.PARAMETER_FILE}")
    print(f"E              : {E_value:.6e} Pa")
    print(f"equations      : {n}   dofs listed: {len(dofs)}   fixed: "
          f"{sum(d['fixed'] for d in dofs)}")
    counts = {}
    for d in dofs:
        key = (d["variable"], d["fixed"])
        counts[key] = counts.get(key, 0) + 1
    for (var, fixed), c in sorted(counts.items()):
        print(f"  {var:<16} {'fixed' if fixed else 'free':<5} {c}")
    print("\nlast equations (Eigen reports 1-based columns after its own reordering):")
    for d in dofs[-2:]:
        print(f"  eq {d['equation_id']:3d} (1-based {d['equation_id'] + 1:3d}): node "
              f"{d['node']:2d} ({d['x']:.3f}, {d['y']:.3f})  {d['variable']:<16} "
              f"{'fixed' if d['fixed'] else 'free'}  K_ii = {d['K_diag']:.3e}")
    print(f"\nsmallest singular values : {', '.join(f'{v:.3e}' for v in s[-5:][::-1])}")
    print(f"largest singular value   : {s[0]:.3e}   condition number: {cond:.3e}")
    print("near-null vector, largest components:")
    for d in sorted(dofs, key=lambda d: -abs(d["null_vector"]))[:12]:
        print(f"  node {d['node']:2d} ({d['x']:.3f}, {d['y']:.3f})  {d['variable']:<16} "
              f"{d['null_vector']:+.4f}")
    by_var = {}
    for d in dofs:
        by_var[d["variable"]] = by_var.get(d["variable"], 0.0) + d["null_vector"] ** 2
    print("share of the near-null vector per variable: " +
          ", ".join(f"{k} {v:.3f}" for k, v in sorted(by_var.items(), key=lambda kv: -kv[1])))
    print(f"wrote {out_csv}")

    analysis.Finalize()


# ------------------------------------------------------------------- re-solves
def solve_all(ids, kind_of, rows_by_id, names, u_ref, E_ref, repeats, out_csv):
    records = []
    for sid in ids:
        r = rows_by_id[sid]
        E = float(r["E_true"])
        kind = kind_of[sid]
        stored = np.array([float(r[f"u_true_{s}"]) for s in names])
        # a good sample is judged against what Phase 1 stored; a bad one against u_ref
        reference = stored if kind == "good" else u_ref * E_ref / E
        tol = GOOD_TOL if kind == "good" else REL_TOL
        for rep in range(1, repeats + 1):
            start = time.perf_counter()
            try:
                u = np.asarray(phase1.run_forward(E, vtk_output_path=None), dtype=float)
                status = "ok"
            except Exception as exc:
                u = np.full(len(names), np.nan)
                status = " ".join(f"failed: {type(exc).__name__}: {exc}".split())[:300]
            rel = np.abs(u / reference - 1.0)
            records.append({
                "sample_id": sid,
                "phase1_kind": kind,
                "repeat": rep,
                "E_true": E,
                "alpha_true": E / E_ref,
                "status": status,
                "seconds": time.perf_counter() - start,
                "reference": "phase1_u_true" if kind == "good" else "u_ref*E_ref/E",
                "tolerance": tol,
                "match": bool(status == "ok" and np.all(rel <= tol)),
                "max_rel_err": float(np.nanmax(rel)) if status == "ok" else float("nan"),
                **{f"u_{s}": float(v) for s, v in zip(names, u)},
                **{f"expected_{s}": float(v) for s, v in zip(names, u_ref * E_ref / E)},
                **{f"phase1_{s}": float(v) for s, v in zip(names, stored)},
            })

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return records


def print_table(ids, records, names, repeats):
    s0 = names[0]      # the tip on the frozen single-sensor layout
    head = f"{'id':>4} {'kind':>6} {'alpha':>8} {'reference':>14}"
    for rep in range(1, repeats + 1):
        head += f" {'solve ' + str(rep):>14} {'ok':>3}"
    print("\n" + head + (f" {'1==2':>5}" if repeats > 1 else f" {'rel err':>9}"))
    for sid in ids:
        rs = [x for x in records if x["sample_id"] == sid]
        ref = rs[0][f"phase1_{s0}"] if rs[0]["phase1_kind"] == "good" \
            else rs[0][f"expected_{s0}"]
        line = f"{sid:4d} {rs[0]['phase1_kind']:>6} {rs[0]['alpha_true']:8.5f} {ref:14.6e}"
        for x in rs:
            line += f" {x[f'u_{s0}']:14.6e} {'y' if x['match'] else 'N':>3}"
        if repeats > 1:
            same = (all(x["status"] == "ok" for x in rs[:2])
                    and abs(rs[0][f"u_{s0}"] / rs[1][f"u_{s0}"] - 1.0) <= REL_TOL)
            line += f" {'y' if same else 'N':>5}"
        else:
            line += f" {rs[0]['max_rel_err']:9.1e}"
        print(line)
        for x in rs:
            if x["status"] != "ok":
                print(f"      repeat {x['repeat']}: {x['status'][:120]}")

    for kind in ("wrong", "failed", "good"):
        sub = [x for x in records if x["phase1_kind"] == kind]
        if sub:
            worst = np.nanmax([x["max_rel_err"] for x in sub]) \
                if any(x["status"] == "ok" for x in sub) else float("nan")
            print(f"{kind:>6}: {sum(x['match'] for x in sub)} of {len(sub)} solves match "
                  f"(tol {sub[0]['tolerance']:g}), worst rel err {worst:.2e}, "
                  f"{sum(x['status'] != 'ok' for x in sub)} failed")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ids", nargs="*", type=int, help="sample ids (default: the 17 bad ones)")
    ap.add_argument("--params", default=phase1.PARAMETER_FILE,
                    help="primal parameter file (default: %(default)s)")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--good", type=int, default=0,
                    help="also re-solve this many valid samples, drawn with a fixed seed")
    ap.add_argument("--good-seed", type=int, default=GOOD_SEED)
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--map-dofs", action="store_true",
                    help="assemble K once (no solve) and report the equation map")
    ap.add_argument("--map-sample", type=int, default=FAILED[0],
                    help="sample whose E is used for --map-dofs (default: %(default)s)")
    args = ap.parse_args()

    # run_forward reads the module constant, so a variant file is swapped in here only
    phase1.PARAMETER_FILE = args.params

    rows, names = load_phase1(AGGREGATE)
    by_id = {int(r["sample_id"]): r for r in rows}
    u_ref = reference_response(rows, names)
    E_ref = phase1.read_reference_young_modulus()

    if args.map_dofs:
        out = args.out if args.out != OUT_CSV else os.path.join(
            OUT_DIR, f"dof_map_{os.path.splitext(os.path.basename(args.params))[0]}.csv")
        map_dofs(float(by_id[args.map_sample]["E_true"]), out)
        return

    bad = args.ids or sorted(WRONG + FAILED)
    missing = [i for i in bad if i not in by_id]
    if missing:
        raise SystemExit(f"sample ids not in {AGGREGATE}: {missing}")
    good = good_ids(rows, names, u_ref, args.good, args.good_seed) if args.good else []
    ids = bad + good
    kind_of = {i: ("failed" if i in FAILED else "good" if i in good else "wrong") for i in ids}

    sensors = phase1.read_sensors(phase1.SENSOR_DATA_PATH)
    if [s["name"] for s in sensors] != names:
        raise SystemExit(f"{phase1.SENSOR_DATA_PATH} sensors {[s['name'] for s in sensors]} "
                         f"do not match the Phase 1 record {names}")

    print(f"parameter file = {args.params}")
    print(f"E_ref = {E_ref:.6e} Pa   u_ref = {u_ref}   OMP_NUM_THREADS = "
          f"{os.environ.get('OMP_NUM_THREADS', 'unset')}")
    if good:
        print(f"good samples (seed {args.good_seed}): {good}")
    print(f"{len(ids)} samples x {args.repeats} solves = {len(ids) * args.repeats} "
          f"forward solves\n")

    records = solve_all(ids, kind_of, by_id, names, u_ref, E_ref, args.repeats, args.out)
    print_table(ids, records, names, args.repeats)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
