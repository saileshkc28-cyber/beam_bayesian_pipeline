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
alpha = data[f"level_{len(q) - 1:02d}"][:, 0]
E_ref = float(data["E_ref"])
mean, std = alpha.mean(), alpha.std(ddof=1)

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.hist(alpha, bins=40, density=True, color="#4878a8", edgecolor="white",
        label=f"posterior samples (n={len(alpha)})")
ax.axvline(mean, color="#1a3a5a", lw=1.8,
           label=rf"mean $\alpha$ = {mean:.4f} $\pm$ {std:.4f}")
for s in (mean - std, mean + std):
    ax.axvline(s, color="#1a3a5a", lw=1.0, ls=":")

truth_file = "../damaged_system/StructuralMaterials.json"
if os.path.exists(truth_file):
    pid = int(data["prop_ids"][0])
    for block in json.load(open(truth_file))["properties"]:
        if block["properties_id"] == pid:
            E_true = block["Material"]["Variables"]["YOUNG_MODULUS"]
            truth = E_true / E_ref
            ax.axvline(truth, color="crimson", ls="--", lw=1.5,
                       label=rf"$\alpha_{{true}}$ = {truth:.4f}")

ax.set_xlabel(r"$\alpha = E/E_{ref}$")
ax.set_ylabel("posterior density")
ax.legend(frameon=False, fontsize=9)
secondary = ax.secondary_xaxis(
    "top", functions=(lambda x: x * E_ref / 1e9, lambda x: x * 1e9 / E_ref))
secondary.set_xlabel("E [GPa]")
fig.tight_layout()
fig.savefig("output/posterior_alpha.png", dpi=150)
print("wrote output/posterior_alpha.png")
