r"""Compare the population posterior of the reference case (GP on forward solves)
with that of the gradient-enhanced case (GP + slope on forward + adjoint solves).

    python plot_posterior_comparison.py --reference <case>\N01 --case <case>\N01_adjoint

Reads only, from both folders: hierarchical\posterior.npz, hierarchical\summary.json,
HierarchicalParameters.json, and surrogate\validation_report.json for the solve
counts (else the training CSV the surrogate was fitted on).
Writes <case>\hierarchical\posterior_comparison.png, never into --reference.
Runs nothing.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# build_gradient_surrogate imports BuildSurrogate, which imports Kratos only
# inside build_forward_model, so this is safe
from build_gradient_surrogate import (CHANGED_KEYS, HIER_DIR, inside, iter_changes,
                                      read_json, resolve_in)
from plot_surrogate import to_float

GPA = 1e9
PARAMS = ("mu_E", "sd_E")
LABELS = {"mu_E": r"$\mu_E$ [GPa]", "sd_E": r"$\sigma_E$ [GPa]"}
N_BINS = 40                      # as plot_population_posterior.py
# colours of hierarchical/plot_population_posterior.py
REF_C = "#4878a8"                # its FINAL_C
GRAD_C = "#c98b3a"               # its 'conservative' design-case colour
# summary.json "data" entries that must agree between the two runs
DATA_KEYS = ("n_observations", "n_sensors", "sensors", "sigma_assumed_m",
             "population_family", "quadrature")


def load_posterior(folder):
    """(n, 2) samples of (mu_E, sd_E) in GPa.
    HierarchicalAnalysis.Finalize stores 'posterior' = the last SMC level, in Pa,
    columns in the order of 'names'. SMC_aCS resamples each level and then moves
    it with aCS chains, so the samples are equally weighted; the file holds no
    weights."""
    path = folder / "hierarchical" / "posterior.npz"
    with np.load(path) as z:
        names = [str(n) for n in z["names"]]
        post = np.asarray(z["posterior"], dtype=float)
    missing = [p for p in PARAMS if p not in names]
    if missing:
        sys.exit(f"{path} has no parameter(s) {', '.join(missing)} (names: {names})")
    return post[:, [names.index(p) for p in PARAMS]] / GPA


def solve_counts(folder):
    """(forward, adjoint, source file) training solves behind the folder's surrogate.
    The validation solves are left out: both cases are scored on the same set."""
    rep_file = folder / "surrogate" / "validation_report.json"
    if rep_file.exists():
        rep = read_json(rep_file)
        # build_gradient_surrogate.py writes the split counts, BuildSurrogate.py
        # only n_training (one forward solve per training point, no adjoint)
        n_f = rep.get("n_forward_solves_training", rep.get("n_training"))
        if n_f is not None:
            return int(n_f), int(rep.get("n_adjoint_solves", 0)), rep_file

    # no report: the training CSV the surrogate was fitted on
    meta = folder / "surrogate" / "response_surrogate.joblib.json"
    train = read_json(meta).get("training_csv") if meta.exists() else None
    cfg_file = folder / "SurrogateParameters.json"
    if not train and cfg_file.exists():
        train = resolve_in(folder, read_json(cfg_file)["output"]["training_csv"])
    if not train or not Path(train).exists():
        sys.exit(f"{folder}: no validation_report.json and no training CSV found")
    with open(train, newline="") as f:
        reader = csv.DictReader(f)
        n_d = sum(c.startswith("dudt_") for c in reader.fieldnames or [])
        e = {to_float(r["E_Pa"]) for r in reader if r.get("status") == "ok"}
    # one forward solve per E, one adjoint per E and sensor (adjoint_training_solves.py)
    return len(e), len(e) * n_d, Path(train)


def solve_label(n_f, n_a):
    return f"{n_f} forward + {n_a} adjoint solves" if n_a else f"{n_f} forward solves"


def looks_like_path(s):
    return isinstance(s, str) and (os.path.isabs(s) or "/" in s or "\\" in s)


def compare_settings(ref, case, s_ref, s_case):
    """(expected, other) differences between the two runs' inputs, each a list of
    (file, key path, reference value, case value)."""
    expected, other = [], []
    f_ref, f_case = ref / "HierarchicalParameters.json", case / "HierarchicalParameters.json"
    if not (f_ref.exists() and f_case.exists()):
        other.append(("HierarchicalParameters.json", ("file",),
                      "present" if f_ref.exists() else "missing",
                      "present" if f_case.exists() else "missing"))
    else:
        # observations file, sigma, quadrature, priors and sampler settings all live here
        for key, a, b in iter_changes(read_json(f_ref), read_json(f_case)):
            # the same file written two ways is no difference; relative paths are
            # relative to hierarchical\, MainHierarchical's working directory
            if (looks_like_path(a) and looks_like_path(b)
                    and resolve_in(HIER_DIR, a) == resolve_in(HIER_DIR, b)):
                continue
            # surrogate_file and output_path are the two keys build_gradient_surrogate.py moves
            (expected if key in CHANGED_KEYS else other).append(
                ("HierarchicalParameters.json", key, a, b))

    # what the runs actually used, as recorded by HierarchicalAnalysis.Finalize
    for k in DATA_KEYS:
        for key, a, b in iter_changes(s_ref["data"].get(k), s_case["data"].get(k),
                                      ("data", k)):
            other.append(("summary.json", key, a, b))
    for k in ("population_family", "parameters"):
        for key, a, b in iter_changes(s_ref.get(k), s_case.get(k), (k,)):
            other.append(("summary.json", key, a, b))
    return expected, other


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, type=Path, help="reference case folder")
    ap.add_argument("--case", required=True, type=Path, help="gradient case folder")
    args = ap.parse_args()

    ref, case = args.reference.resolve(), args.case.resolve()
    png = case / "hierarchical" / "posterior_comparison.png"
    if inside(png, ref):
        sys.exit(f"refusing to write {png}: it lies inside --reference {ref}")

    post = {"ref": load_posterior(ref), "case": load_posterior(case)}
    summ = {"ref": read_json(ref / "hierarchical" / "summary.json"),
            "case": read_json(case / "hierarchical" / "summary.json")}
    counts = {"ref": solve_counts(ref), "case": solve_counts(case)}
    label = {w: solve_label(*counts[w][:2]) for w in counts}

    # mean and sd as Finalize computes them (sd with ddof=1)
    stats = {w: {p: (post[w][:, k].mean(), post[w][:, k].std(ddof=1))
                 for k, p in enumerate(PARAMS)} for w in post}

    # the samples must be the ones summary.json was written from
    for w, folder in (("ref", ref), ("case", case)):
        for p in PARAMS:
            s = summ[w]["posterior"][p]
            if not np.allclose(stats[w][p], (s["mean"] / GPA, s["sd"] / GPA), rtol=1e-9):
                print(f"WARNING {folder.name}: {p} in posterior.npz does not match "
                      f"summary.json -- one of the two files is stale")

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for k, (p, ax) in enumerate(zip(PARAMS, axes)):
        # identical bin edges for both runs
        edges = np.histogram_bin_edges(np.concatenate([post["ref"][:, k],
                                                       post["case"][:, k]]), bins=N_BINS)
        for w, color, ls in (("ref", REF_C, "-"), ("case", GRAD_C, "--")):
            m, s = stats[w][p]
            ax.hist(post[w][:, k], bins=edges, density=True, histtype="step",
                    color=color, ls=ls, lw=1.6,
                    label=f"{label[w]}\nmean = {m:.4f} $\\pm$ {s:.4f} GPa")
            ax.axvline(m, color=color, ls=ls, lw=1.2)
        ax.set_xlabel(LABELS[p])
        ax.set_ylabel("posterior density")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.25)
    fig.suptitle(f"Population posterior:  {ref.name}  vs  {case.name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ----------------------------------------------------------------- table
    print(f"reference  {ref}\n  {label['ref']}  (from {counts['ref'][2]})")
    print(f"gradient   {case}\n  {label['case']}  (from {counts['case'][2]})")
    print(f"\n{'param':<7}{'':<36}{'mean [GPa]':>12}{'sd [GPa]':>12}")
    for p in PARAMS:
        (m_r, s_r), (m_c, s_c) = stats["ref"][p], stats["case"][p]
        print(f"{p:<7}{'reference':<36}{m_r:>12.4f}{s_r:>12.4f}")
        print(f"{p:<7}{'gradient':<36}{m_c:>12.4f}{s_c:>12.4f}")
        print(f"{p:<7}{'difference (gradient - reference)':<36}{m_c - m_r:>+12.4f}"
              f"{s_c - s_r:>+12.4f}")
        print(f"{p:<7}{'difference / reference posterior sd':<36}{(m_c - m_r) / s_r:>+12.3f}"
              f"{(s_c - s_r) / s_r:>+12.3f}")

    # ------------------------------------------------------- settings check
    expected, other = compare_settings(ref, case, summ["ref"], summ["case"])
    print("\nsettings check (HierarchicalParameters.json, summary.json)")
    for f, key, a, b in expected:
        print(f"  expected  {f}: {'.'.join(map(str, key))}\n"
              f"            reference {a}\n            gradient  {b}")
    for f, key, a, b in other:
        print(f"  DIFFERENT {f}: {'.'.join(map(str, key))}\n"
              f"            reference {a}\n            gradient  {b}")
    if not other:
        print("  no other differences: same observations file, sigma, quadrature, "
              "priors and sampler settings")
    print(f"\n-> {png}")


if __name__ == "__main__":
    main()
