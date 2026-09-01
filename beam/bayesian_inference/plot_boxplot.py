"""Post-processing only. Box plots of alpha from the saved SMC posterior.

Writes:
  output/posterior_boxplot.png   one box per SMC level (+ zoom) with the mean
                                 labelled, and a horizontal q = 1 panel below
                                 showing quartiles against mean +/- 1 SD
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = np.load("output/smc_levels.npz")
q = data["q"]
E_ref = float(np.atleast_1d(data["E_ref"])[0])
levels = [data[f"level_{i:02d}"][:, 0] for i in range(len(q))]
alpha = levels[-1]

truth = None
truth_file = "../damaged_system/StructuralMaterials.json"
if os.path.exists(truth_file):
    pid = int(data["prop_ids"][0])
    for block in json.load(open(truth_file))["properties"]:
        if block["properties_id"] == pid:
            truth = block["Material"]["Variables"]["YOUNG_MODULUS"] / E_ref

n = len(levels)
colors = plt.cm.viridis(np.linspace(0.15, 0.9, n))
colors[-1] = matplotlib.colors.to_rgba("#4878a8")

mean_l = np.array([a.mean() for a in levels])
std_l = np.array([a.std(ddof=1) for a in levels])
q1_l = np.array([np.percentile(a, 25) for a in levels])
med_l = np.array([np.median(a) for a in levels])
q3_l = np.array([np.percentile(a, 75) for a in levels])
iqr_l = q3_l - q1_l

DARK = "#1a3a5a"
# matplotlib >= 3.10 renamed vert=False to orientation="horizontal"
HORIZ = ({"orientation": "horizontal"}
         if tuple(int(v) for v in matplotlib.__version__.split(".")[:2]) >= (3, 10)
         else {"vert": False})

BOX_KW = dict(patch_artist=True, showmeans=True, whis=1.5,
              medianprops=dict(color=DARK, lw=1.8),
              meanprops=dict(marker="D", ms=6, mfc="white", mec=DARK, mew=1.2),
              flierprops=dict(marker=".", ms=3, mfc="0.45", mec="none", alpha=0.45),
              whiskerprops=dict(color=DARK, lw=1.0),
              capprops=dict(color=DARK, lw=1.0))


def style_boxes(bp, idx):
    for patch, i in zip(bp["boxes"], idx):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.75)
        patch.set_edgecolor(DARK)
        patch.set_linewidth(0.8)


def gpa_axis(ax, which):
    f = (lambda v: v * E_ref / 1e9, lambda v: v * 1e9 / E_ref)
    if which == "y":
        ax.secondary_yaxis("right", functions=f).set_ylabel("E [GPa]", fontsize=9)
    else:
        ax.secondary_xaxis("top", functions=f).set_xlabel("E [GPa]", fontsize=9)


# ============================================ figure 1: one box per SMC level
def draw_levels(ax, idx):
    idx = list(idx)
    bp = ax.boxplot([levels[i] for i in idx], positions=np.arange(len(idx)),
                    widths=0.55, **BOX_KW)
    style_boxes(bp, idx)
    # numeric mean above each box, at the top whisker cap
    caps = bp["caps"][1::2]
    for k, i in enumerate(idx):
        top = caps[k].get_ydata()[0]
        ax.annotate(rf"$\bar\alpha$ = {mean_l[i]:.4f}", (k, top),
                    textcoords="offset points", xytext=(0, 8), ha="center",
                    fontsize=7.5, color=DARK, rotation=90, va="bottom")
    if truth is not None:
        ax.axhline(truth, color="crimson", ls="--", lw=1.5, zorder=0,
                   label=rf"$\alpha_{{true}}$ = {truth:.4f}")
    ax.plot([], [], marker="D", ls="none", ms=6, mfc="white", mec=DARK,
            label=r"mean $\bar\alpha$")
    ax.set_xticks(np.arange(len(idx)))
    ax.set_xticklabels([f"{q[i]:.4f}" for i in idx], rotation=45, fontsize=8)
    ax.set_xlabel(r"tempering parameter $q$")
    ax.set_ylabel(r"$\alpha = E/E_{ref}$")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.margins(y=0.05)
    lo, hi = ax.get_ylim()          # extra headroom for the rotated mean labels
    ax.set_ylim(lo, hi + 0.30 * (hi - lo))
    gpa_axis(ax, "y")


fig = plt.figure(figsize=(13.5, 9.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.52],
                      width_ratios=[1.7, 1], hspace=0.85, wspace=0.45)

ax1 = fig.add_subplot(gs[0, 0])
draw_levels(ax1, range(n))
ax1.set_title(rf"$\alpha$ per SMC level  |  IQR contracts "
              rf"{iqr_l[0] / iqr_l[-1]:.0f}$\times$", fontsize=11, pad=10)
ax1.legend(frameon=False, fontsize=9, loc="lower left")

nz = min(3, n)
idx2 = list(range(n - nz, n))
ax2 = fig.add_subplot(gs[0, 1])
draw_levels(ax2, idx2)
ax2.set_title(f"final {nz} levels (zoom)", fontsize=11, pad=10)

# ------------------------- bottom row: horizontal box for the q = 1 level
m, s = mean_l[-1], std_l[-1]
ax = fig.add_subplot(gs[1, :])

ax.axvspan(m - s, m + s, color=DARK, alpha=0.07, zorder=0)
ax.errorbar(m, 0.55, xerr=s, fmt="D", ms=7, mfc="white", mec=DARK, mew=1.4,
            ecolor=DARK, elinewidth=1.6, capsize=6)
ax.annotate(rf"$\bar\alpha$ = {m:.4f}   SD = {s:.4f}", (m, 0.55),
            textcoords="offset points", xytext=(0, 13), ha="center",
            fontsize=8.5, color=DARK)

bp = ax.boxplot([alpha], positions=[1.15], widths=0.35, **BOX_KW, **HORIZ)
style_boxes(bp, [n - 1])

if truth is not None:
    ax.axvline(truth, color="crimson", ls="--", lw=1.5, zorder=0,
               label=rf"$\alpha_{{true}}$ = {truth:.4f}")

for val, name in ((q1_l[-1], "Q1"), (med_l[-1], "median"), (q3_l[-1], "Q3")):
    ax.annotate(f"{name}\n{val:.4f}", (val, 1.36), ha="center", va="bottom",
                fontsize=7.5, color=DARK)

ax.set_yticks([0.55, 1.15])
ax.set_yticklabels([r"mean $\pm$ 1 SD", "quartiles"], fontsize=9)
ax.set_ylim(0.30, 1.65)
ax.set_xlabel(r"$\alpha = E/E_{ref}$")
ax.set_title(rf"posterior at $q$ = {q[-1]:.4f}   (n = {len(alpha)}, "
             rf"IQR = {iqr_l[-1]:.4f})", fontsize=11, pad=34)
ax.grid(True, axis="x", alpha=0.25)
ax.set_axisbelow(True)
if truth is not None:
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
gpa_axis(ax, "x")

fig.savefig("output/posterior_boxplot.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote output/posterior_boxplot.png\n")

# =========================================================== summary table
print(f"{'level':>5} {'q':>8} {'mean':>9} {'SD':>9} {'Q1':>9} {'median':>9} "
      f"{'Q3':>9} {'IQR':>9} {'fliers':>7}")
for i, a in enumerate(levels):
    lo, hi = q1_l[i] - 1.5 * iqr_l[i], q3_l[i] + 1.5 * iqr_l[i]
    nf = int(((a < lo) | (a > hi)).sum())
    print(f"{i:>5} {q[i]:>8.4f} {mean_l[i]:>9.4f} {std_l[i]:>9.4f} {q1_l[i]:>9.4f} "
          f"{med_l[i]:>9.4f} {q3_l[i]:>9.4f} {iqr_l[i]:>9.4f} {nf:>7d}")