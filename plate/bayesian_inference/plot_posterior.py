"""Post-processing only. Reads the saved posterior and, if available, the Phase 1
truth record. Kept out of MainBayesian.py so the inference never touches the truth."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = np.load("output/smc_levels.npz")
q = data["q"]
posterior = np.atleast_2d(data[f"level_{len(q) - 1:02d}"])
refs = np.atleast_1d(data["E_ref"])
prop_ids = np.atleast_1d(data["prop_ids"])
n_zones = posterior.shape[1]

truth = {}
truth_file = "../damaged_system/StructuralMaterials.json"
if os.path.exists(truth_file):
    blocks = {b["properties_id"]: b for b in json.load(open(truth_file))["properties"]}
    for z in range(n_zones):
        block = blocks.get(int(prop_ids[z]))
        if block and "YOUNG_MODULUS" in block["Material"].get("Variables", {}):
            truth[z] = block["Material"]["Variables"]["YOUNG_MODULUS"] / refs[z]

# ---------------------------------------------------------------- marginals
rows = int(np.ceil(n_zones / 2))
fig, axes = plt.subplots(rows, 2, figsize=(11, 3.4 * rows), squeeze=False)
for z in range(n_zones):
    ax = axes[z // 2][z % 2]
    a = posterior[:, z]
    mean, std = a.mean(), a.std(ddof=1)
    ax.hist(a, bins=40, density=True, color="#4878a8", edgecolor="white")
    ax.axvline(mean, color="#1a3a5a", lw=1.8,
               label=rf"mean = {mean:.4f} $\pm$ {std:.4f}")
    for s in (mean - std, mean + std):
        ax.axvline(s, color="#1a3a5a", lw=1.0, ls=":")
    if z in truth:
        ax.axvline(truth[z], color="crimson", ls="--", lw=1.5,
                   label=rf"true = {truth[z]:.4f}")
    ax.set_title(rf"zone {z + 1}   ($E_{{ref}}$ = {refs[z] / 1e9:.1f} GPa)")
    ax.set_xlabel(rf"$\alpha_{{{z + 1}}}$")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8)
for k in range(n_zones, rows * 2):
    axes[k // 2][k % 2].axis("off")
fig.tight_layout()
fig.savefig("output/posterior_alpha.png", dpi=150)

# ------------------------------------------------------- correlation matrix
if n_zones > 1:
    corr = np.corrcoef(posterior, rowvar=False)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [rf"$\alpha_{{{z + 1}}}$" for z in range(n_zones)]
    ax.set_xticks(range(n_zones), labels)
    ax.set_yticks(range(n_zones), labels)
    for i in range(n_zones):
        for j in range(n_zones):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(corr[i, j]) > 0.5 else "black", fontsize=9)
    ax.set_title("posterior correlation")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig("output/posterior_correlation.png", dpi=150)

    print("posterior correlation matrix:")
    print(np.array2string(corr, precision=3, suppress_small=True))

for z in range(n_zones):
    a = posterior[:, z]
    line = (f"zone {z + 1}: alpha = {a.mean():.4f} +/- {a.std(ddof=1):.4f}"
            f"   E = {a.mean() * refs[z] / 1e9:.2f} GPa")
    if z in truth:
        line += f"   (true alpha = {truth[z]:.4f})"
    print(line)
print("wrote output/posterior_alpha.png"
      + (" and output/posterior_correlation.png" if n_zones > 1 else ""))
