r"""Collects the noise-seed sweep and draws the three figures. Read-only; safe to re-run.

    python collect_seed_sweep.py                                # defaults to Noise_Seed
    python collect_seed_sweep.py D:\...\Analysis\Noise_Seed

Writes results.csv, seed_gaussian.png and seed_boxplot_E.png next to the run folders.
"""
import csv
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ARCHIVE = Path(sys.argv[1] if len(sys.argv) > 1
               else r"D:\KratosProjects\MCMC\Analysis\Noise_Seed")

E_TRUE_GPA = 206.9
SIGMA_2PCT = 3.788097e-08
NOISE_FRACTION = 0.02

BOX = "#5b7c99"
EDGE = "#1f3b57"
TRUTH = "#d62728"
COLD = "#2a78d6"
WARM = "#d85a30"

FIELDS = ["run", "label", "seed", "z", "noise_fraction_data", "sigma_assumed", "e_ref",
          "alpha_mean", "alpha_std", "alpha_p2.5", "alpha_p97.5",
          "E_mean_GPa", "E_std_GPa", "E_p2.5_GPa", "E_p97.5_GPa", "covers_truth",
          "n_levels", "n_forward_solves", "wall_time_s", "logcE"]


def posterior_samples(folder):
    for path in sorted(glob.glob(str(folder / "*.npz"))):
        with np.load(path) as z:
            keys = [k for k in z.files if k.startswith("level_")]
            if keys:
                return np.asarray(z[sorted(keys)[-1]])[:, 0]
    return None


SKIPPED = []


def collect(folder):
    summary_file = folder / "summary.json"
    if not summary_file.exists():
        info_file = folder / "run_info.json"
        why = "no run_info.json either -- Phase 1 or the sweep aborted here"
        if info_file.exists():
            code = json.loads(info_file.read_text()).get("returncode")
            why = (f"Phase 2 exited with {code}" if code
                   else "Phase 2 reported success but wrote nothing")
        SKIPPED.append((folder.name, why))
        return None
    summary = json.loads(summary_file.read_text())
    zone = summary["zones"][0]
    info = json.loads((folder / "run_info.json").read_text())

    e_ref = info["e_ref"] / 1e9
    lo, hi = zone["alpha_p2.5"] * e_ref, zone["alpha_p97.5"] * e_ref

    return {
        "run": folder.name,
        "label": info["label"],
        "seed": info["seed"],
        "z": info["z"],
        "noise_fraction_data": info["noise_fraction_data"],
        "sigma_assumed": info["sigma_assumed"],
        "e_ref": info["e_ref"],
        "alpha_mean": zone["alpha_mean"],
        "alpha_std": zone["alpha_std"],
        "alpha_p2.5": zone["alpha_p2.5"],
        "alpha_p97.5": zone["alpha_p97.5"],
        "E_mean_GPa": zone["alpha_mean"] * e_ref,
        "E_std_GPa": zone["alpha_std"] * e_ref,
        "E_p2.5_GPa": lo,
        "E_p97.5_GPa": hi,
        "covers_truth": lo <= E_TRUE_GPA <= hi,
        "n_levels": len(summary["tempering_q"]),
        "n_forward_solves": summary["n_forward_solves"],
        "wall_time_s": info.get("wall_time_s"),
        "logcE": summary["logcE"],
        "_samples": posterior_samples(folder),
    }


def save(fig, out):
    with open(str(out), "wb") as f:
        fig.savefig(f, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_gaussian(rows, out):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.4))
    fig.subplots_adjust(hspace=0.38)

    t = np.linspace(-3.2, 3.2, 400)
    ax1.plot(t, np.exp(-t ** 2 / 2) / np.sqrt(2 * np.pi), color="0.45", lw=1.6)
    ax1.axvline(0, color="0.75", ls="--", lw=1)
    for r in rows:
        z = r["z"]
        c = "0.25" if r["noise_fraction_data"] == 0 else (WARM if z > 0 else COLD)
        h = np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
        ax1.vlines(z, 0, h, color=c, lw=1.1)
        ax1.plot(z, h, "o", color=c, ms=8,
                 mfc="white" if r["noise_fraction_data"] == 0 else c)
        ax1.annotate(f"{r['E_mean_GPa']:.1f}", (z, h), textcoords="offset points",
                     xytext=(0, 11), ha="center", fontsize=9, color=c)
    ax1.set_xlim(-3.2, 3.2)
    ax1.set_ylim(0, 0.47)
    ax1.set_xlabel(r"Phase 1 noise draw  $z$   [$\sigma$ units]")
    ax1.set_ylabel("standard normal density")
    ax1.set_title("Where each run's noise draw landed   (label = recovered E, GPa)",
                  fontsize=12)
    ax1.legend(handles=[
        Line2D([], [], marker="o", color="none", mfc=COLD, mec=COLD, ms=8, label="softer  (z < 0)"),
        Line2D([], [], marker="o", color="none", mfc=WARM, mec=WARM, ms=8, label="stiffer  (z > 0)"),
        Line2D([], [], marker="o", color="none", mfc="white", mec="0.25", ms=8, label="no noise"),
    ], loc="upper left", fontsize=9, frameon=False)

    z = np.array([r["z"] for r in rows])
    E = np.array([r["E_mean_GPa"] for r in rows])
    err = np.vstack([E - np.array([r["E_p2.5_GPa"] for r in rows]),
                     np.array([r["E_p97.5_GPa"] for r in rows]) - E])
    ax2.plot(t, E_TRUE_GPA * (1 + NOISE_FRACTION * t), color="0.55", ls="--", lw=1.3,
             label=rf"analytic  $E \approx E_{{true}}(1 + {NOISE_FRACTION:g}\,z)$")
    ax2.axhline(E_TRUE_GPA, color=TRUTH, ls="--", lw=1.6,
                label=f"truth {E_TRUE_GPA} GPa")
    ax2.errorbar(z, E, yerr=err, fmt="o", color=EDGE, ms=7, capsize=4, lw=1.3,
                 label="posterior mean, 95% CI")
    ax2.set_xlim(-3.2, 3.2)
    ax2.set_xlabel(r"Phase 1 noise draw  $z$   [$\sigma$ units]")
    ax2.set_ylabel("recovered E  [GPa]")
    ax2.set_title("Recovered E follows the noise draw, not a method bias", fontsize=12)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=9, framealpha=0.9)

    save(fig, out)


def figure_boxplot(rows, out):
    have = all(r["_samples"] is not None for r in rows)
    rng = np.random.default_rng(0)
    data, pos = [], np.arange(len(rows))
    for r in rows:
        e_ref = r["e_ref"] / 1e9
        s = (r["_samples"] * e_ref if r["_samples"] is not None
             else rng.normal(r["E_mean_GPa"], r["E_std_GPa"], 4000))
        data.append(s)

    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.boxplot(data, positions=pos, widths=0.55, vert=False, patch_artist=True,
               showfliers=False, whis=(2.5, 97.5),
               medianprops=dict(color=EDGE, lw=2),
               boxprops=dict(facecolor=BOX, edgecolor=EDGE, alpha=0.75, lw=1.2),
               whiskerprops=dict(color=EDGE, lw=1.2),
               capprops=dict(color=EDGE, lw=1.2))
    for p, d in zip(pos, data):
        ax.plot(np.mean(d), p, "D", mfc="white", mec=EDGE, ms=7, zorder=5)
    ax.axvline(E_TRUE_GPA, color=TRUTH, ls="--", lw=1.8, zorder=1)

    lo = min(np.percentile(d, 2.5) for d in data)
    hi = max(np.percentile(d, 97.5) for d in data)
    span = hi - lo
    ax.set_xlim(lo - 0.09 * span, hi + 0.42 * span)
    for p, r in zip(pos, rows):
        bias = 100.0 * (r["E_mean_GPa"] - E_TRUE_GPA) / E_TRUE_GPA
        mark = "" if r["covers_truth"] else "   *"
        ax.text(hi + 0.03 * span, p,
                f"{r['E_mean_GPa']:.2f} GPa   ({bias:+.2f}%){mark}",
                va="center", ha="left", fontsize=9, color=EDGE)

    ax.set_yticks(pos)
    ax.set_yticklabels([f"{r['label']}   seed {r['seed']}" if r["noise_fraction_data"]
                        else f"{r['label']}" for r in rows])
    ax.set_ylabel("Phase 1 noise draw")
    ax.set_xlabel("E  [GPa]")
    ax.set_title(f"Posterior E across noise seeds   |   Phase 2 sigma fixed at "
                 f"{SIGMA_2PCT:.6e} (matched 2%)   |   * = 95% CI misses truth",
                 fontsize=12, pad=14)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(handles=[
        Line2D([], [], color=TRUTH, ls="--", lw=1.8, label=f"E true = {E_TRUE_GPA} GPa"),
        Line2D([], [], marker="D", color="none", mfc="white", mec=EDGE, ms=7, label="mean"),
    ], loc="lower right", fontsize=9, framealpha=0.9)

    note = ("boxes from posterior samples" if have
            else "boxes Gaussian-implied from summary.json (mean, SD)")
    fig.text(0.5, -0.02, note, ha="center", fontsize=8, color="grey", style="italic")
    save(fig, out)


def main():
    if not ARCHIVE.exists():
        sys.exit(f"no such archive: {ARCHIVE}")

    folders = [f for f in sorted(ARCHIVE.iterdir()) if f.is_dir()]
    rows = [r for r in (collect(f) for f in folders) if r]
    if SKIPPED:
        print(f"*** {len(SKIPPED)} of {len(folders)} run folders have no result "
              f"and are NOT in the table below ***")
        for name, why in SKIPPED:
            print(f"    {name:<20} {why}")
        print()
    if not rows:
        sys.exit(f"no summary.json found under {ARCHIVE}")
    rows.sort(key=lambda r: r["z"])

    sig = {r["sigma_assumed"] for r in rows}
    if len(sig) > 1:
        print("WARNING: sigma_assumed varies across runs -- this sweep should hold it fixed")
    elif abs(sig.pop() - SIGMA_2PCT) / SIGMA_2PCT > 1e-3:
        print("WARNING: sigma_assumed is not the matched 2% value")

    out = ARCHIVE / "results.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({k: r[k] for k in FIELDS} for r in rows)

    header = (f"{'run':<12}{'seed':>11}{'z':>8}{'alpha':>9}{'std':>9}"
              f"{'E [GPa]':>10}{'bias%':>8}{'95% CI [GPa]':>20}{'cov':>5}"
              f"{'lvls':>6}{'solves':>8}{'logcE':>9}{'time_s':>8}")
    print(header)
    print("-" * len(header))
    for r in rows:
        ci = f"[{r['E_p2.5_GPa']:.2f},{r['E_p97.5_GPa']:.2f}]"
        t = "" if r["wall_time_s"] is None else format(r["wall_time_s"], ">8.0f")
        print(f"{r['label']:<12}{r['seed']:>11}{r['z']:>8.3f}{r['alpha_mean']:>9.4f}"
              f"{r['alpha_std']:>9.4f}{r['E_mean_GPa']:>10.2f}"
              f"{100 * (r['E_mean_GPa'] - E_TRUE_GPA) / E_TRUE_GPA:>8.2f}{ci:>20}"
              f"{'y' if r['covers_truth'] else 'N':>5}"
              f"{r['n_levels']:>6}{r['n_forward_solves']:>8}{r['logcE']:>9.3f}{t}")

    E = np.array([r["E_mean_GPa"] for r in rows])
    cov = sum(r["covers_truth"] for r in rows)
    zero = next((r for r in rows if r["noise_fraction_data"] == 0), None)
    print(f"\nE across seeds: mean {E.mean():.2f} GPa, spread {E.max() - E.min():.2f} GPa "
          f"({100 * (E.max() - E.min()) / E_TRUE_GPA:.2f}% of truth)")
    if zero:
        print(f"noise-free reference: {zero['E_mean_GPa']:.2f} GPa "
              f"({100 * (zero['E_mean_GPa'] - E_TRUE_GPA) / E_TRUE_GPA:+.2f}%), "
              f"SD {zero['E_std_GPa']:.2f} GPa")
    print(f"coverage: {cov}/{len(rows)} of the 95% intervals contain {E_TRUE_GPA} GPa")

    figure_gaussian(rows, ARCHIVE / "seed_gaussian.png")
    figure_boxplot(rows, ARCHIVE / "seed_boxplot_E.png")
    print(f"\nwrote {out}\n      {ARCHIVE / 'seed_gaussian.png'}"
          f"\n      {ARCHIVE / 'seed_boxplot_E.png'}")


if __name__ == "__main__":
    main()
