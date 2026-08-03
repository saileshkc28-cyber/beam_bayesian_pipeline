"""Convert output/smc_levels.npz to an Excel workbook (one sheet per tempering level
plus a summary). Run after MainBayesian.py."""
import json
import numpy as np
import pandas as pd

data = np.load("output/smc_levels.npz")
q = data["q"]
E_ref = float(data["E_ref"])
n_levels = len(q)
posterior = data[f"level_{n_levels - 1:02d}"]

with pd.ExcelWriter("output/posterior.xlsx", engine="openpyxl") as writer:

    summary = json.load(open("output/summary.json"))
    pd.DataFrame([{"quantity": k, "value": v} for k, v in summary.items()
                  if not isinstance(v, list)]).to_excel(
        writer, sheet_name="summary", index=False)

    for lev in range(n_levels):
        samples = data[f"level_{lev:02d}"]
        cols = {f"alpha_{z + 1}": samples[:, z] for z in range(samples.shape[1])}
        for z in range(samples.shape[1]):
            cols[f"E_{z + 1}_Pa"] = samples[:, z] * E_ref
        df = pd.DataFrame(cols)
        df.insert(0, "particle", np.arange(1, len(df) + 1))
        df.to_excel(writer, sheet_name=f"level_{lev:02d}_q{q[lev]:.3f}"[:31], index=False)

    stats = []
    for z in range(posterior.shape[1]):
        a = posterior[:, z]
        stats.append({"zone": z + 1, "alpha_mean": a.mean(), "alpha_std": a.std(ddof=1),
                      "alpha_p2.5": np.percentile(a, 2.5),
                      "alpha_p97.5": np.percentile(a, 97.5),
                      "E_mean_Pa": a.mean() * E_ref, "E_std_Pa": a.std(ddof=1) * E_ref})
    pd.DataFrame(stats).to_excel(writer, sheet_name="posterior_stats", index=False)

print(f"wrote output/posterior.xlsx  ({n_levels} levels, "
      f"{len(posterior)} particles, {posterior.shape[1]} zone(s))")
