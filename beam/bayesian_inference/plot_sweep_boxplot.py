r"""
Horizontal box plots of the posterior alpha for any sweep (load, prior, E_ref, sigma).

    python plot_sweep_boxplot.py D:\KratosProjects\MCMC\Analysis\Prior

Auto-detects which column was swept. Uses raw posterior samples from
output_*/*.npz when available, otherwise Gaussian-implied boxes from results.csv.
Warns if sigma_assumed is not the matched 2% value.
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "results.csv")
OUT = os.path.join(HERE, "sweep_boxplot.png")

ALPHA_TRUE = 1.0
LOAD_REF = 1000.0
SIGMA_2PCT = 3.788097e-08

BOX = "#5b7c99"
EDGE = "#1f3b57"
TRUTH = "#d62728"
TXT = "#2c4a63"

CANDIDATES = {
    "prior_parameters": "prior",
    "load_modulus": "assumed load [N/m]",
    "e_ref": "E_ref [Pa]",
    "sigma_assumed": "assumed sigma",
}


def detect_swept(df):
    for col, axis_label in CANDIDATES.items():
        if col in df.columns and df[col].astype(str).nunique() > 1:
            return col, axis_label
    return "run", "run"


def load_samples(run_dir):
    for path in sorted(glob.glob(os.path.join(HERE, str(run_dir), "*.npz"))):
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
    rng = np.random.default_rng(0)
    return np.clip(rng.normal(mean, sd, n), lo, hi)


def draw(ax, data, positions):
    ax.boxplot(
        data, positions=positions, widths=0.55, vert=False,
        patch_artist=True, showfliers=False, whis=(2.5, 97.5),
        medianprops=dict(color=EDGE, lw=2),
        boxprops=dict(facecolor=BOX, edgecolor=EDGE, alpha=0.75, lw=1.2),
        whiskerprops=dict(color=EDGE, lw=1.2),
        capprops=dict(color=EDGE, lw=1.2),
    )
    for pos, d in zip(positions, data):
        ax.plot(np.mean(d), pos, "D", mfc="white", mec=EDGE, ms=7, zorder=5)


def main():
    if not os.path.isfile(CSV):
        sys.exit(f"no results.csv in {HERE}\n"
                 f"pass the sweep folder:  python {os.path.basename(__file__)} <folder>")

    df = pd.read_csv(CSV)
    swept, axis_label = detect_swept(df)
    if pd.api.types.is_numeric_dtype(df[swept]):
        df = df.sort_values(swept)
    df = df.reset_index(drop=True)
    e_ref = float(df["e_ref"].iloc[0])

    sig = df["sigma_assumed"].unique()
    if len(sig) == 1 and abs(sig[0] - SIGMA_2PCT) / SIGMA_2PCT < 1e-3:
        sig_note = f"sigma = {sig[0]:.6e}  (matched 2%)"
    elif len(sig) == 1:
        sig_note = f"sigma = {sig[0]:.6e}  ({100 * sig[0] / SIGMA_2PCT * 0.02:.2f}% -- NOT the matched 2%)"
        print(f"WARNING: {sig_note}")
    else:
        sig_note = "sigma varies across runs"

    samples, source = [], []
    for _, r in df.iterrows():
        s = load_samples(r["run"])
        source.append("npz" if s is not None else "csv")
        if s is None:
            s = gaussian_box(r["alpha_mean"], r["alpha_std"], r["alpha_p2.5"], r["alpha_p97.5"])
        samples.append(s)

    pos = np.arange(len(df))
    if "label" in df.columns and df["label"].notna().all():
        labels = [str(v) for v in df["label"]]
    else:
        labels = [str(v) for v in df[swept]]

    load_varies = swept == "load_modulus"
    noise_varies = swept == "sigma_assumed"
    eref_varies = swept == "e_ref"
    E_UNDAMAGED = 206.9e9
    if noise_varies:
        labels = [f"{2.0 * v / SIGMA_2PCT:.1f}%" for v in df[swept]]
    if eref_varies:
        labels = [f"{v / 1e9:.1f} GPa" for v in df[swept]]
    two_panel = load_varies or noise_varies or eref_varies
    fig, axes = plt.subplots(2 if two_panel else 1, 1,
                             figsize=(11, 8.5 if two_panel else 5.2), squeeze=False)
    axes = axes[:, 0]
    fig.subplots_adjust(hspace=0.42, top=0.86)

    ax = axes[0]
    draw(ax, samples, pos)
    if not eref_varies:
        ax.axvline(ALPHA_TRUE, color=TRUTH, ls="--", lw=1.8, zorder=1)

    lo = min(np.percentile(d, 2.5) for d in samples)
    hi = max(np.percentile(d, 97.5) for d in samples)
    span = hi - lo
    ax.set_xlim(lo - (0.14 if eref_varies else 0.09) * span, hi + 0.50 * span)
    for p, (_, r) in zip(pos, df.iterrows()):
        if eref_varies:
            label_txt = rf"$\bar\alpha$ = {r['alpha_mean']:.4f}"
        else:
            bias = 100.0 * (r["alpha_mean"] - ALPHA_TRUE)
            label_txt = rf"$\bar\alpha$ = {r['alpha_mean']:.4f}   ({bias:+.1f}%)"
        ax.text(hi + 0.04 * span, p, label_txt,
                va="center", ha="left", fontsize=9, color=TXT)

    ax.set_yticks(pos)
    ax.set_yticklabels(labels)
    ax.set_ylabel("assumed noise (% of |u|max)" if noise_varies
                  else ("reference modulus $E_{ref}$" if eref_varies else axis_label))
    ax.set_xlabel(r"$\alpha = E / E_{ref}$")
    title_note = "true noise in the data = 2.0%" if noise_varies else sig_note
    if eref_varies:
        title_note = r"$\alpha$ is scaled by $E_{ref}$ -- see recovered E below"
    ax.set_title(rf"Posterior $\alpha$ vs. "
                 rf"{'assumed noise' if noise_varies else ('$E_{ref}$' if eref_varies else axis_label)}"
                 rf"    |    {title_note}",
                 fontsize=12, pad=32)
    ax.grid(axis="x", alpha=0.25)
    handles = [Line2D([], [], marker="D", color="none", mfc="white", mec=EDGE, ms=7,
                      label=r"mean $\bar\alpha$")]
    if not eref_varies:
        handles.insert(0, Line2D([], [], color=TRUTH, ls="--", lw=1.8,
                                 label=r"$\alpha_{true}$ = 1.0000"))
    ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.9)

    top = ax.twiny()
    top.set_xlim(*[v * e_ref / 1e9 for v in ax.get_xlim()])
    top.set_xlabel("E [GPa]")

    if load_varies:
        scale = LOAD_REF / df[swept].to_numpy()
        scaled = [s * f for s, f in zip(samples, scale)]
        collapse = np.array([np.mean(s) for s in scaled])
        ax2 = axes[1]
        draw(ax2, scaled, pos)
        ax2.axvline(ALPHA_TRUE, color=TRUTH, ls="--", lw=1.8, zorder=1)
        ax2.axvline(collapse.mean(), color=EDGE, ls=":", lw=1.6, zorder=1)
        lo2 = min(np.percentile(d, 2.5) for d in scaled)
        hi2 = max(np.percentile(d, 97.5) for d in scaled)
        span2 = hi2 - lo2
        ax2.set_xlim(lo2 - 0.06 * span2, hi2 + 0.16 * span2)
        for p, m in zip(pos, collapse):
            ax2.text(hi2 + 0.02 * span2, p, f"{m:.4f}",
                     va="center", ha="left", fontsize=9, color=TXT)
        ax2.set_yticks(pos)
        ax2.set_yticklabels(labels)
        ax2.set_ylabel(axis_label)
        ax2.set_xlabel(r"$\alpha \cdot F_{true} / F_{assumed}$")
        ax2.set_title(f"Load-corrected: collapse to {collapse.mean():.4f} "
                      f"(spread {100 * np.ptp(collapse) / collapse.mean():.2f}%)  -- F/E degeneracy",
                      fontsize=12)
        ax2.grid(axis="x", alpha=0.25)
        ax2.legend(handles=[
            Line2D([], [], color=TRUTH, ls="--", lw=1.8, label=r"$\alpha_{true}$ = 1.0000"),
            Line2D([], [], color=EDGE, ls=":", lw=1.6,
                   label=f"collapse mean = {collapse.mean():.4f}"),
        ], loc="upper left", fontsize=9, framealpha=0.9)

    if noise_varies:
        ax2 = axes[1]
        sig = df[swept].to_numpy(dtype=float)
        sd = np.array([np.std(d, ddof=1) for d in samples])
        ratio = sig / SIGMA_2PCT
        ax2.plot(ratio, sd, "o-", color=EDGE, ms=7, mfc=BOX, lw=1.5,
                 label="posterior SD")
        base = sd[np.argmin(np.abs(ratio - 1.0))]
        ax2.plot(ratio, base * ratio, "--", color=TRUTH, lw=1.6,
                 label=r"linear in $\sigma$ (through the matched run)")
        ax2.axvline(1.0, color="grey", ls=":", lw=1.2)
        for x, y in zip(ratio, sd):
            ax2.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                         xytext=(0, 9), ha="center", fontsize=8, color=TXT)
        ax2.set_xlabel(r"assumed $\sigma$ / true $\sigma$")
        ax2.set_ylabel(r"posterior SD of $\alpha$")
        dev = 100 * np.max(np.abs(sd / (base * ratio) - 1.0))
        ax2.set_title(f"Posterior width vs. assumed noise "
                      f"(max departure from linear: {dev:.1f}%)", fontsize=12)
        ax2.grid(alpha=0.25)
        ax2.legend(loc="upper left", fontsize=9, framealpha=0.9)

    if eref_varies:
        ax2 = axes[1]
        e_samples = [s * v for s, v in zip(samples, df[swept].to_numpy(dtype=float))]
        e_samples = [s / 1e9 for s in e_samples]
        draw(ax2, e_samples, pos)
        means = np.array([np.mean(s) for s in e_samples])
        ax2.axvline(E_UNDAMAGED / 1e9, color=TRUTH, ls="--", lw=1.8, zorder=1)
        ax2.axvline(means.mean(), color=EDGE, ls=":", lw=1.6, zorder=1)
        lo2 = min(np.percentile(d, 2.5) for d in e_samples)
        hi2 = max(np.percentile(d, 97.5) for d in e_samples)
        span2 = hi2 - lo2
        ax2.set_xlim(lo2 - 0.06 * span2, hi2 + 0.20 * span2)
        for p_, m in zip(pos, means):
            ax2.text(hi2 + 0.02 * span2, p_, f"{m:.2f}",
                     va="center", ha="left", fontsize=9, color=TXT)
        ax2.set_yticks(pos)
        ax2.set_yticklabels(labels)
        ax2.set_ylabel("reference modulus $E_{ref}$")
        ax2.set_xlabel(r"recovered $E = \alpha \cdot E_{ref}$  [GPa]")
        ax2.set_title(f"Recovered E: mean {means.mean():.2f} GPa, spread "
                      f"{100 * np.ptp(means) / means.mean():.2f}%  -- "
                      f"E_ref is a parameterisation, not an assumption", fontsize=12)
        ax2.grid(axis="x", alpha=0.25)
        ax2.legend(handles=[
            Line2D([], [], color=TRUTH, ls="--", lw=1.8,
                   label=f"undamaged E = {E_UNDAMAGED / 1e9:.1f} GPa"),
            Line2D([], [], color=EDGE, ls=":", lw=1.6,
                   label=f"recovered mean = {means.mean():.2f} GPa"),
        ], loc="upper left", fontsize=9, framealpha=0.9)

    note = ("boxes from posterior samples" if "npz" in source else
            "boxes Gaussian-implied from results.csv (mean, SD); whiskers = reported 95% CI")
    fig.text(0.5, -0.04 if len(axes) == 1 else 0.015, note, ha="center", fontsize=8, color="grey", style="italic")

    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}   swept: {swept}   [{', '.join(source)}]")


if __name__ == "__main__":
    main()
