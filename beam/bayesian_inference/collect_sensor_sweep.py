r"""Collects the sensor-count sweep and draws two figures. Read-only; safe to re-run.

    python collect_sensor_sweep.py                              # defaults to Sensors
    python collect_sensor_sweep.py D:\...\Analysis\Sensors

The interesting quantity is the effective sensor count. With one zone and a linear
model, u_i(alpha) = u_i_ref / alpha, so the Fisher information is

    I(alpha) = sum_i u_i_ref^2 / (alpha^4 sigma^2)

and the posterior SD should fall as 1 / sqrt(N_eff) with

    N_eff(k) = sum_{i<=k} u_i^2 / u_tip^2

which is read straight from each run's noise_model.json -- no analytic beam theory.
Under a single absolute sigma, root sensors have tiny u and contribute almost
nothing, so N_eff saturates well below k. The naive 1/sqrt(k) line is drawn for
contrast.

Writes results.csv, sensor_information.png and sensor_boxplot_E.png.
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
               else r"D:\KratosProjects\MCMC\Analysis\Sensors")

E_TRUE_GPA = 206.9
SIGMA_2PCT = 3.788097e-08

BOX = "#5b7c99"
EDGE = "#1f3b57"
TRUTH = "#d62728"
PRED = "#2a78d6"
NAIVE = "#d85a30"

FIELDS = ["run", "label", "n_sensors", "stations", "n_eff", "sigma_assumed", "e_ref",
          "alpha_mean", "alpha_std", "alpha_p2.5", "alpha_p97.5",
          "E_mean_GPa", "E_std_GPa", "E_p2.5_GPa", "E_p97.5_GPa", "covers_truth",
          "sd_ratio", "sd_pred_ratio", "n_levels", "n_forward_solves",
          "wall_time_s", "logcE"]

SKIPPED = []


def posterior_samples(folder):
    for path in sorted(glob.glob(str(folder / "*.npz"))):
        with np.load(path) as z:
            keys = [k for k in z.files if k.startswith("level_")]
            if keys:
                return np.asarray(z[sorted(keys)[-1]])[:, 0]
    return None


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
    info = json.loads((folder / "run_info.json").read_text())
    zone = summary["zones"][0]

    u = np.asarray(info["u_true"], float)
    n_eff = float((u ** 2).sum() / u[0] ** 2)

    e_ref = info["e_ref"] / 1e9
    lo, hi = zone["alpha_p2.5"] * e_ref, zone["alpha_p97.5"] * e_ref

    return {
        "run": folder.name,
        "label": info["label"],
        "n_sensors": info["n_sensors"],
        "stations": " ".join(format(x, ".1f") for x in info["stations"]),
        "n_eff": n_eff,
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
        "_u": u,
    }


def save(fig, out):
    with open(str(out), "wb") as f:
        fig.savefig(f, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_information(rows, out):
    k = np.array([r["n_sensors"] for r in rows], float)
    n_eff = np.array([r["n_eff"] for r in rows])
    sd = np.array([r["alpha_std"] for r in rows])
    sd0 = sd[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.4))
    fig.subplots_adjust(hspace=0.36)

    ax1.plot(k, k, color=NAIVE, ls="--", lw=1.4, label=r"naive  $N_{eff} = k$")
    ax1.plot(k, n_eff, "o-", color=EDGE, ms=7, lw=1.6, label=r"$N_{eff}$ from $u_{true}$")
    for ki, ne in zip(k, n_eff):
        ax1.annotate(f"{ne:.2f}", (ki, ne), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8, color=EDGE)
    ax1.set_xticks(k)
    ax1.set_xlabel("sensors  k   (tip first, then inward)")
    ax1.set_ylabel(r"effective sensor count  $N_{eff}$")
    ax1.set_title(r"Information saturates: $N_{eff} = \sum u_i^2 / u_{tip}^2$ "
                  "under one absolute sigma", fontsize=12)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=9, framealpha=0.9)

    ax2.plot(k, sd0 / np.sqrt(k), "s--", color=NAIVE, ms=6, lw=1.4,
             label=r"naive  $SD_1/\sqrt{k}$")
    ax2.plot(k, sd0 / np.sqrt(n_eff), "^--", color=PRED, ms=7, lw=1.5,
             label=r"predicted  $SD_1/\sqrt{N_{eff}}$")
    ax2.plot(k, sd, "o-", color=EDGE, ms=7, lw=1.8, label="posterior SD (sampled)")
    ax2.set_xticks(k)
    ax2.set_xlabel("sensors  k")
    ax2.set_ylabel(r"posterior SD of $\alpha$")
    ax2.set_title("Sampled SD against the Fisher prediction", fontsize=12)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=9, framealpha=0.9)

    resid = 100.0 * (sd - sd0 / np.sqrt(n_eff)) / (sd0 / np.sqrt(n_eff))
    fig.text(0.5, 0.015,
             f"max deviation from the Fisher prediction: {np.abs(resid).max():.1f}%",
             ha="center", fontsize=8, color="grey", style="italic")
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

    fig, ax = plt.subplots(figsize=(11, 6.2))
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
    ax.set_xlim(lo - 0.09 * span, hi + 0.46 * span)
    for p, r in zip(pos, rows):
        bias = 100.0 * (r["E_mean_GPa"] - E_TRUE_GPA) / E_TRUE_GPA
        mark = "" if r["covers_truth"] else "   *"
        ax.text(hi + 0.03 * span, p,
                f"{r['E_mean_GPa']:.2f} GPa   ({bias:+.2f}%){mark}",
                va="center", ha="left", fontsize=9, color=EDGE)

    ax.set_yticks(pos)
    ax.set_yticklabels([f"{r['n_sensors']} sensor{'s' if r['n_sensors'] > 1 else ''}"
                        f"   $N_{{eff}}$ = {r['n_eff']:.2f}" for r in rows])
    ax.set_ylabel("sensor count")
    ax.set_xlabel("E  [GPa]")
    ax.set_title(f"Posterior E vs. sensor count   |   sigma fixed at {SIGMA_2PCT:.6e} "
                 f"(matched 2%)   |   * = 95% CI misses truth", fontsize=12, pad=14)
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
    rows.sort(key=lambda r: r["n_sensors"])

    sig = {r["sigma_assumed"] for r in rows}
    if len(sig) > 1:
        print("WARNING: sigma_assumed varies across runs -- this sweep should hold it fixed")
    elif abs(sig.pop() - SIGMA_2PCT) / SIGMA_2PCT > 1e-3:
        print("WARNING: sigma_assumed is not the matched 2% value")

    base = rows[0]
    if base["n_sensors"] != 1:
        print("WARNING: no N=1 run -- SD ratios are relative to the smallest run present")
    for r in rows:
        r["sd_ratio"] = r["alpha_std"] / base["alpha_std"]
        r["sd_pred_ratio"] = float(np.sqrt(base["n_eff"] / r["n_eff"]))

    out = ARCHIVE / "results.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({k: r[k] for k in FIELDS} for r in rows)

    header = (f"{'run':<7}{'k':>4}{'N_eff':>8}{'alpha':>9}{'std':>9}{'SD/SD1':>9}"
              f"{'pred':>8}{'E [GPa]':>10}{'bias%':>8}{'95% CI [GPa]':>20}{'cov':>5}"
              f"{'lvls':>6}{'solves':>8}{'logcE':>9}{'time_s':>8}")
    print(header)
    print("-" * len(header))
    for r in rows:
        ci = f"[{r['E_p2.5_GPa']:.2f},{r['E_p97.5_GPa']:.2f}]"
        t = "" if r["wall_time_s"] is None else format(r["wall_time_s"], ">8.0f")
        print(f"{r['label']:<7}{r['n_sensors']:>4}{r['n_eff']:>8.2f}{r['alpha_mean']:>9.4f}"
              f"{r['alpha_std']:>9.4f}{r['sd_ratio']:>9.3f}{r['sd_pred_ratio']:>8.3f}"
              f"{r['E_mean_GPa']:>10.2f}"
              f"{100 * (r['E_mean_GPa'] - E_TRUE_GPA) / E_TRUE_GPA:>8.2f}{ci:>20}"
              f"{'y' if r['covers_truth'] else 'N':>5}"
              f"{r['n_levels']:>6}{r['n_forward_solves']:>8}{r['logcE']:>9.3f}{t}")

    last = rows[-1]
    print(f"\nN_eff saturates at {last['n_eff']:.2f} with {last['n_sensors']} sensors "
          f"(naive would be {last['n_sensors']})")
    print(f"SD shrinks {1 / last['sd_ratio']:.2f}x, Fisher predicts "
          f"{1 / last['sd_pred_ratio']:.2f}x, naive 1/sqrt(k) predicts "
          f"{np.sqrt(last['n_sensors']):.2f}x")
    cov = sum(r["covers_truth"] for r in rows)
    print(f"coverage: {cov}/{len(rows)} of the 95% intervals contain {E_TRUE_GPA} GPa")

    figure_information(rows, ARCHIVE / "sensor_information.png")
    figure_boxplot(rows, ARCHIVE / "sensor_boxplot_E.png")
    print(f"\nwrote {out}\n      {ARCHIVE / 'sensor_information.png'}"
          f"\n      {ARCHIVE / 'sensor_boxplot_E.png'}")


if __name__ == "__main__":
    main()
