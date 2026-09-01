r"""
Horizontal box plots of the posterior alpha for the assumed-load sweep.

Run from D:\KratosProjects\MCMC\Analysis\Load

    python plot_load_sweep.py

Uses raw posterior samples from output_<load>/*.npz when available,
otherwise builds Gaussian-implied boxes from results.csv.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "results.csv")
OUT = os.path.join(HERE, "load_sweep_boxplot.png")
ALPHA_TRUE = 1.0
LOAD_REF = 1000.0

BOX = "#5b7c99"
EDGE = "#1f3b57"
TRUTH = "#d62728"
TXT = "#2c4a63"


def load_samples(run_dir):
    """Return the 1-D posterior alpha samples for a run, or None."""
    for path in sorted(glob.glob(os.path.join(HERE, run_dir, "*.npz"))):
        with np.load(path) as z:
            for key in ("samples", "posterior", "alpha", "theta", "levels", "samplesU", "samplesX"):
                if key not in z:
                    continue
                a = np.asarray(z[key])
                if a.ndim == 3:
                    a = a[-1]
                if a.ndim == 2:
                    a = a[:, 0] if a.shape[0] >= a.shape[1] else a[0]
                a = a.ravel()
                if a.size > 50:
                    return a
    return None


def gaussian_box(mean, sd, lo, hi, n=4000):
    """Synthetic sample matching mean/sd, clipped to the reported 95% CI."""
    rng = np.random.default_rng(0)
    s = rng.normal(mean, sd, n)
    return np.clip(s, lo, hi)


def draw(ax, data, positions, widths=0.55):
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=widths,
        vert=False,
        patch_artist=True,
        showfliers=False,
        whis=(2.5, 97.5),
        medianprops=dict(color=EDGE, lw=2),
        boxprops=dict(facecolor=BOX, edgecolor=EDGE, alpha=0.75, lw=1.2),
        whiskerprops=dict(color=EDGE, lw=1.2),
        capprops=dict(color=EDGE, lw=1.2),
    )
    for pos, d in zip(positions, data):
        ax.plot(np.mean(d), pos, "D", mfc="white", mec=EDGE, ms=7, zorder=5)
    return bp


def main():
    df = pd.read_csv(CSV).sort_values("load_modulus").reset_index(drop=True)
    e_ref = float(df["e_ref"].iloc[0])

    samples, source = [], []
    for _, r in df.iterrows():
        s = load_samples(str(r["run"]))
        if s is None:
            s = gaussian_box(r["alpha_mean"], r["alpha_std"], r["alpha_p2.5"], r["alpha_p97.5"])
            source.append("csv")
        else:
            source.append("npz")
        samples.append(s)

    scale = LOAD_REF / df["load_modulus"].to_numpy()
    scaled = [s * f for s, f in zip(samples, scale)]
    pos = np.arange(len(df))
    labels = [f"{v:.0f} N/m" for v in df["load_modulus"]]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharey=True)
    fig.subplots_adjust(hspace=0.42, top=0.88)

    # --- panel 1: raw posterior alpha -------------------------------------
    ax = axes[0]
    draw(ax, samples, pos)
    ax.axvline(ALPHA_TRUE, color=TRUTH, ls="--", lw=1.8, zorder=1)
    for p, (_, r) in zip(pos, df.iterrows()):
        bias = 100.0 * (r["alpha_mean"] - ALPHA_TRUE)
        ax.text(
            r["alpha_p97.5"] + 0.012, p,
            rf"$\bar\alpha$ = {r['alpha_mean']:.4f}   ({bias:+.1f}%)",
            va="center", ha="left", fontsize=9, color=TXT,
        )
    ax.set_yticks(pos)
    ax.set_yticklabels(labels)
    ax.set_ylabel("assumed load in Phase 2")
    ax.set_xlabel(r"$\alpha = E / E_{ref}$")
    lo = min(np.percentile(d, 2.5) for d in samples)
    hi = max(np.percentile(d, 97.5) for d in samples)
    ax.set_xlim(lo - 0.09 * (hi - lo), hi + 0.50 * (hi - lo))
    ax.set_title(
        r"Posterior $\alpha$ vs. assumed load    |    true load = 1000 N/m,  $\alpha_{true}$ = 1.0",
        fontsize=12, pad=32,
    )
    ax.grid(axis="x", alpha=0.25)

    top = ax.twiny()
    top.set_xlim(*[v * e_ref / 1e9 for v in ax.get_xlim()])
    top.set_xlabel("E [GPa]")

    ax.legend(
        handles=[
            Line2D([], [], color=TRUTH, ls="--", lw=1.8, label=r"$\alpha_{true}$ = 1.0000"),
            Line2D([], [], marker="D", color="none", mfc="white", mec=EDGE, ms=7,
                   label=r"mean $\bar\alpha$"),
        ],
        loc="upper left", fontsize=9, framealpha=0.9,
    )

    # --- panel 2: load-corrected alpha ------------------------------------
    ax2 = axes[1]
    draw(ax2, scaled, pos)
    ax2.axvline(ALPHA_TRUE, color=TRUTH, ls="--", lw=1.8, zorder=1)
    collapse = np.array([np.mean(s) for s in scaled])
    ax2.axvline(collapse.mean(), color=EDGE, ls=":", lw=1.6, zorder=1)
    hi2 = max(np.percentile(d, 97.5) for d in scaled)
    lo2 = min(np.percentile(d, 2.5) for d in scaled)
    for p, m in zip(pos, collapse):
        ax2.text(hi2 + 0.02 * (hi2 - lo2), p, f"{m:.4f}",
                 va="center", ha="left", fontsize=9, color=TXT)
    ax2.set_yticks(pos)
    ax2.set_yticklabels(labels)
    ax2.set_ylabel("assumed load in Phase 2")
    ax2.set_xlabel(r"$\alpha \cdot F_{true} / F_{assumed}$")
    ax2.set_xlim(lo2 - 0.06 * (hi2 - lo2), hi2 + 0.16 * (hi2 - lo2))
    ax2.set_title(
        f"Load-corrected: all five collapse to {collapse.mean():.4f} "
        f"(spread {100 * np.ptp(collapse) / collapse.mean():.2f}%)  —  F/E degeneracy",
        fontsize=12,
    )
    ax2.grid(axis="x", alpha=0.25)
    ax2.legend(
        handles=[
            Line2D([], [], color=TRUTH, ls="--", lw=1.8, label=r"$\alpha_{true}$ = 1.0000"),
            Line2D([], [], color=EDGE, ls=":", lw=1.6,
                   label=f"collapse mean = {collapse.mean():.4f}"),
        ],
        loc="upper left", fontsize=9, framealpha=0.9,
    )

    note = "boxes from posterior samples" if "npz" in source else \
        "boxes Gaussian-implied from results.csv (mean, SD); whiskers = reported 95% CI"
    fig.text(0.5, 0.015, note, ha="center", fontsize=8, color="grey", style="italic")

    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}  [{', '.join(source)}]")


if __name__ == "__main__":
    main()
