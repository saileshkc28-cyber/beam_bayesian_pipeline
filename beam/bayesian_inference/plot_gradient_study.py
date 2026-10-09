r"""Plot the training-subset study of gradient_surrogate_study.py: max validation
error / sigma against the number of training points, per method, with the gate.

    python plot_gradient_study.py --source <case>\N01 --case <case>\N01_adjoint

Reads only, from --source: SurrogateParameters.json (accuracy gate) and
surrogate\gradient_surrogate_study.csv. Writes <case>\surrogate\gradient_study.png,
never into --source. Runs nothing.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# build_gradient_surrogate imports BuildSurrogate, which imports Kratos only
# inside build_forward_model, so this is safe
from build_gradient_surrogate import inside
from plot_surrogate import to_float

STUDY_CSV = "gradient_surrogate_study.csv"   # written by gradient_surrogate_study.py
# (key, legend label, fitted on slopes too); keys as in gradient_surrogate_study.MODELS
METHODS = (("gp", "GP (values)", False),
           ("pchip", "PCHIP (values)", False),
           ("hermite", "Hermite (values + slopes)", True),
           ("gp_slope", "GP + slope", True))
KEYS = [k for k, _, _ in METHODS] + ["analytic"]
MARKERS = {"gp": "o", "pchip": "s", "hermite": "^", "gp_slope": "D"}
COLORS = {"gp": "C0", "pchip": "C1", "hermite": "C3", "gp_slope": "C2"}


def col(key):
    """Column name in the study CSV, as gradient_surrogate_study.py writes it."""
    return f"{key}_max_error_over_sigma"


def read_study(path):
    """Rows of the study CSV: row type, n, and max error / sigma per method
    (NaN where the cell is empty, i.e. the method was not run for that row)."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in ["row", "n"] + [col(k) for k in KEYS]
                   if c not in (reader.fieldnames or [])]
        if missing:
            sys.exit(f"{path} has no column(s) {', '.join(missing)}")
        rows = list(reader)
    return [{"row": r["row"], "n": int(r["n"]),
             **{k: to_float(r[col(k)]) if r[col(k)] != "" else float("nan") for k in KEYS}}
            for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path, help="reference case folder")
    ap.add_argument("--case", required=True, type=Path, help="gradient case folder")
    args = ap.parse_args()

    src, case = args.source.resolve(), args.case.resolve()
    png = case / "surrogate" / "gradient_study.png"
    if inside(png, src):
        sys.exit(f"refusing to write {png}: it lies inside --source {src}")
    if not png.parent.is_dir():
        sys.exit(f"{png.parent} does not exist -- run build_gradient_surrogate.py first")

    with open(src / "SurrogateParameters.json") as f:
        gate = json.load(f)["accuracy_gate"]["max_error_over_sigma"]
    rows = read_study(src / "surrogate" / STUDY_CSV)
    subset = sorted((r for r in rows if r["row"] == "subset"), key=lambda r: r["n"])
    adjoint = [r for r in rows if r["row"] == "adjoint"]
    analytic = [r for r in rows if r["row"] == "analytic"]
    if not subset:
        sys.exit(f"{src / 'surrogate' / STUDY_CSV} has no 'subset' rows")
    ns = np.array([r["n"] for r in subset])

    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    # subset rows: values-only methods solid with filled markers,
    # slope methods dashed with open markers
    for key, label, slopes in METHODS:
        y = np.array([r[key] for r in subset])
        y = np.where(y > 0, y, np.nan)   # log axis: failed fits (NaN) and zeros left out
        ax.plot(ns, y, "--" if slopes else "-", marker=MARKERS[key], ms=6, lw=1.4,
                mew=1.4, color=COLORS[key], mfc="white" if slopes else COLORS[key],
                label=label)

    # adjoint row: Hermite and GP + slope refitted with the Kratos adjoint slopes
    for r in adjoint:
        for key in ("hermite", "gp_slope"):
            if np.isfinite(r[key]) and r[key] > 0:
                ax.plot(r["n"], r[key], "*", ms=16, color=COLORS[key], mec="black",
                        mew=0.8, zorder=5)

    ax.axhline(gate, ls="--", color="0.3", lw=1.1)
    ax.text(0.99, gate, f"gate = {gate:g}", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=9, color="0.3")

    ax.set_xscale("log")
    ax.set_yscale("log")
    # ticks at the subset sizes only, as plain integers
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("training points")
    ax.set_ylabel(r"max validation error / $\sigma$")
    ax.set_title("Response surrogate accuracy against training points")

    handles, labels = ax.get_legend_handles_labels()
    if adjoint:
        # one neutral legend entry for the stars of both slope methods
        handles.append(Line2D([], [], ls="none", marker="*", ms=12, color="0.6",
                              mec="black", mew=0.8))
        labels.append("slopes from Kratos adjoint")
    ax.legend(handles, labels, frameon=False, fontsize=9)
    ax.grid(alpha=0.3)

    fig.text(0.01, 0.01, "slope methods (dashed, open markers): one training point = "
             "1 forward + 1 adjoint solve per sensor", fontsize=8, color="0.3")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(png, dpi=160)

    # smallest subset size below the gate (strict, as gradient_surrogate_study.gate_word)
    print(f"gate: max error / sigma < {gate:g}   ({src / 'SurrogateParameters.json'})")
    print("smallest n that passes the gate, subset rows:")
    for key, label, _ in METHODS:
        ok = [r["n"] for r in subset if r[key] < gate]   # NaN compares False
        print(f"  {label:<28}" + (f"n = {min(ok)}" if ok else f"none up to n = {ns.max()}"))
    for r in adjoint:
        print(f"adjoint row, n = {r['n']}:  Hermite {r['hermite']:.3e}  "
              f"GP + slope {r['gp_slope']:.3e}")
    for r in analytic:
        word = ("n/a" if not np.isfinite(r["analytic"])
                else "PASS" if r["analytic"] < gate else "FAIL")
        print(f"analytic row (not plotted), n = {r['n']}:  {r['analytic']:.3e}  {word}")
    print(f"-> {png}")


if __name__ == "__main__":
    main()
