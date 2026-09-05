"""Post-processing only. Reads the Phase 1 sampling record and, if available, the
distribution metadata. Kept out of MainKratos_phase1.py so plots can be restyled
without re-running any Kratos solves.

Deliberately mirrors plot_posterior.py: same colours, same alpha axis with a
secondary E [GPa] axis on top, same mean +- 1 std annotation, so a Phase 1 figure
and a Phase 2 posterior figure can sit side by side in the thesis.

The distinction the two scripts draw:
  plot_posterior.py   spread WITHIN one posterior   -> uncertainty about one alpha
  plot_phase1.py      spread BETWEEN realizations   -> the population being propagated

Writes two figures:
  <root>/phase1_alpha.png    the E / alpha population on its own
  <root>/phase1_panels.png   xi, alpha and u side by side

    python plot_phase1.py
    python plot_phase1.py phase1_distribution_runs
"""
import csv
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = sys.argv[1] if len(sys.argv) > 1 else "phase1_distribution_runs"

# ------------------------------------------------------------------ read the record
aggregate = os.path.join(root, "phase1_samples.csv")
if not os.path.isfile(aggregate):
    raise SystemExit(f"no Phase 1 record found: {aggregate}")

with open(aggregate, newline="") as f:
    rows = [r for r in csv.DictReader(f) if r.get("status", "").strip().strip('"') == "ok"]
if len(rows) < 2:
    raise SystemExit(f"only {len(rows)} valid samples in {aggregate}")

u_true_columns = [c for c in rows[0] if c.startswith("u_true_")]
sensor_names = [c[len("u_true_"):] for c in u_true_columns]

xi = np.array([float(r["xi"]) for r in rows])
E = np.array([float(r["E_true"]) for r in rows])
alpha = np.array([float(r["alpha_true"]) for r in rows])
u = np.array([[float(r[c]) for c in u_true_columns] for r in rows])
u_hat = np.array([[float(r[f"u_hat_{s}"]) for s in sensor_names] for r in rows])

# target distribution and reference stiffness, from any sample's metadata
E_ref = E[0] / alpha[0]
target_mean = target_sd = sigma = None
for r in rows:
    path = os.path.join(root, f"sample_{int(r['sample_id']):04d}",
                        "measurement_metadata.json")
    if os.path.isfile(path):
        meta = json.load(open(path))
        E_ref = float(meta.get("E_ref", E_ref))
        target_mean = float(meta["E_distribution_mean"]) / E_ref
        target_sd = float(meta["E_distribution_sd"]) / E_ref
        sigma = float(meta.get("sigma", 0.0))
        break

n = len(rows)
colors = plt.cm.viridis(np.linspace(0.15, 0.9, 3))
main_color = matplotlib.colors.to_rgba("#4878a8")   # the population keeps the blue
NAVY = "#1a3a5a"


def normal_pdf(x, mean, sd):
    return np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * np.sqrt(2.0 * np.pi))


def gpa_axis(ax):
    secondary = ax.secondary_xaxis(
        "top", functions=(lambda x: x * E_ref / 1e9, lambda x: x * 1e9 / E_ref))
    secondary.set_xlabel("E [GPa]", fontsize=9)


def draw_population(ax, data, color, label, mean_symbol, target=None, bins=None,
                    override=None):
    """One histogram in the plot_posterior.py idiom: navy mean, dotted +- 1 std,
    crimson dashed reference where one exists.

    override = (mean, sd) takes the canonical whole-run values instead of recomputing."""
    bins = bins if bins is not None else min(40, max(6, n // 3))
    mean, std = override if override is not None else (data.mean(), data.std(ddof=1))
    ax.hist(data, bins=bins, density=True, color=color, edgecolor="white",
            label=f"n = {n}")
    ax.axvline(mean, color=NAVY, lw=1.8,
               label=rf"mean {mean_symbol} = {mean:.4f} $\pm$ {std:.4f}")
    for s in (mean - std, mean + std):
        ax.axvline(s, color=NAVY, lw=1.0, ls=":")
    if target is not None:
        t_mean, t_sd = target
        grid = np.linspace(data.min(), data.max(), 400)
        ax.plot(grid, normal_pdf(grid, t_mean, t_sd), color="crimson", ls="--", lw=1.5,
                label=rf"target $\mathcal{{N}}$({t_mean:.3f}, {t_sd:.3f})")
    ax.set_xlabel(label)
    return mean, std


# ==================================================== figure 1: the E population
fig, ax = plt.subplots(figsize=(7, 4.2))
mean_a, std_a = draw_population(
    ax, alpha, main_color, r"$\alpha = E/E_{ref}$", r"$\alpha$",
    target=(target_mean, target_sd) if target_sd else None)
ax.set_ylabel("population density")
ax.legend(frameon=False, fontsize=9)
gpa_axis(ax)
fig.savefig(os.path.join(root, "phase1_alpha.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {os.path.join(root, 'phase1_alpha.png')}")

# ======================================== figure 2: inputs, response, convergence
fig = plt.figure(figsize=(6.6 * 3, 4.6))
gs = fig.add_gridspec(1, 3, wspace=0.24)

# --- the chain xi -> E -> u
ax = fig.add_subplot(gs[0, 0])
mean_x, std_x = draw_population(ax, xi, colors[0], r"$\xi$", r"$\xi$", target=(0.0, 1.0))
ax.set_ylabel("density")
ax.set_title(r"standard normal input $\xi \sim \mathcal{N}(0,1)$", fontsize=11)
ax.legend(frameon=False, fontsize=8, loc="upper right")

ax = fig.add_subplot(gs[0, 1])
draw_population(ax, alpha, colors[1], r"$\alpha = E/E_{ref}$", r"$\alpha$",
                target=(target_mean, target_sd) if target_sd else None)
ax.set_ylabel("density")
ax.set_title(r"stiffness $E = \mu_E + \sigma_E\,\xi$", fontsize=11, pad=32)
ax.legend(frameon=False, fontsize=8, loc="upper right")
gpa_axis(ax)

ax = fig.add_subplot(gs[0, 2])
u_main = u[:, 0] * 1e6
skew = float(((u_main - u_main.mean()) ** 3).mean() / u_main.std() ** 3)

# the whole-run u statistics come from the canonical summary when it exists, so the
# figure label and the collapsed CSV always quote the same number
canonical = os.path.join(root, "phase1_response_summary.json")
u_stats = None
if os.path.isfile(canonical):
    s = json.load(open(canonical))
    u_stats = (s["u_mean"][0] * 1e6, s["u_sd"][0] * 1e6)
mean_u, std_u = draw_population(ax, u_main, colors[2],
                                rf"$u$ [{sensor_names[0]}]  [$\mu$m]", "$u$",
                                override=u_stats)
ax.set_ylabel("density")
ax.set_title(rf"response $u = \mathcal{{G}}(E)$   skew = {skew:+.3f}", fontsize=11)
ax.legend(frameon=False, fontsize=8, loc="upper right")

fig.savefig(os.path.join(root, "phase1_panels.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {os.path.join(root, 'phase1_panels.png')}\n")

# ------------------------------------------------------------------ console record
print(f"{'quantity':>12} {'mean':>14} {'sd':>14} {'cov':>8} {'min':>14} {'max':>14}")
for label, data in ((r"xi", xi), ("alpha", alpha), ("E [GPa]", E / 1e9),
                    (f"u [um]", u_main)):
    m, s = data.mean(), data.std(ddof=1)
    print(f"{label:>12} {m:>14.6f} {s:>14.6f} {s / abs(m):>8.4f} "
          f"{data.min():>14.6f} {data.max():>14.6f}")

if target_sd:
    print(f"\ntarget alpha : mean {target_mean:.6f}   sd {target_sd:.6f}")
    print(f"empirical    : mean {alpha.mean():.6f}   sd {alpha.std(ddof=1):.6f}   "
          f"({100 * (alpha.std(ddof=1) / target_sd - 1):+.2f}% on sd)")
if sigma:
    snr = sigma / abs(u[:, 0]).mean()
    print(f"\nmeasurement noise sigma = {sigma:.6e}  "
          f"({100 * snr:.2f}% of the mean response)")
    print(f"between-realization sd of u = {u[:, 0].std(ddof=1):.6e}  "
          f"({u[:, 0].std(ddof=1) / sigma:.1f}x the noise)")
if len(sensor_names) > 1:
    print(f"\nsensors plotted: {sensor_names[0]} (of {len(sensor_names)}: "
          f"{', '.join(sensor_names)})")
