r"""Builds one table from any sweep archive. Read-only; safe to re-run.

    python collect_sweep.py                          # defaults to Prior
    python collect_sweep.py D:\...\Analysis\Load

Writes results.csv next to the run folders and prints the table.
Layout adapts: load sweeps get the scaled column, prior sweeps get
prior width and the Occam check.
"""
import ast
import csv
import json
import math
import sys
from pathlib import Path

ARCHIVE = Path(sys.argv[1] if len(sys.argv) > 1
               else r"D:\KratosProjects\MCMC\Analysis\E_ref")

SIGMA_2PCT = 3.788097e-08
ALPHA_TRUE = 1.0

# Undamaged modulus the Phase 1 data was generated from. alpha is E_recovered
# divided by the Phase 2 GUESS, so alpha equals the damage ratio only when the
# guess happens to equal this value. The damage column below always divides by
# this, so it stays comparable across sweeps.
E_UNDAMAGED = 206.9e9

FIELDS = ["run", "label", "load_modulus", "e_ref", "sigma_assumed", "prior_type",
          "prior_parameters", "alpha_mean", "alpha_std", "alpha_p2.5",
          "alpha_p97.5", "alpha_scaled", "E_mean_Pa", "E_std_Pa", "n_levels",
          "n_forward_solves", "wall_time_s", "logcE"]


def collect(folder):
    summary_file = folder / "summary.json"
    if not summary_file.exists():
        return None
    summary = json.loads(summary_file.read_text())
    zone = summary["zones"][0]

    info = {}
    info_file = folder / "run_info.json"
    if info_file.exists():
        info = json.loads(info_file.read_text())

    load = info.get("load_modulus")
    params = info.get("prior_parameters")
    e_mean = zone.get("E_mean") or zone.get("E_mean_Pa")
    e_ref = info.get("e_ref") or info.get("reference_value")
    if e_ref is None and e_mean and zone["alpha_mean"]:
        e_ref = e_mean / zone["alpha_mean"]

    return {
        "run": folder.name,
        "label": info.get("label", folder.name.replace("output_", "")),
        "load_modulus": load,
        "e_ref": e_ref,
        "sigma_assumed": info.get("sigma_assumed"),
        "prior_type": info.get("prior_type"),
        "prior_parameters": ",".join(str(p) for p in params) if params else None,
        "alpha_mean": zone["alpha_mean"],
        "alpha_std": zone["alpha_std"],
        "alpha_p2.5": zone["alpha_p2.5"],
        "alpha_p97.5": zone["alpha_p97.5"],
        # u ~ F/alpha, so alpha should track the assumed load; flat if that holds
        "alpha_scaled": (zone["alpha_mean"] * 1000.0 / load) if load else None,
        "E_mean_Pa": e_mean,
        "E_std_Pa": zone.get("E_std") or zone.get("E_std_Pa"),
        "n_levels": len(summary["tempering_q"]),
        "n_forward_solves": summary["n_forward_solves"],
        "wall_time_s": info.get("wall_time_s"),
        "logcE": summary["logcE"],
    }


def prior_text(row):
    t = row.get("prior_type") or ""
    raw = row.get("prior_parameters")
    if not raw:
        return row["label"]
    p = [float(v) for v in str(raw).split(",")]
    if t == "uniform":
        return f"uniform[{p[0]:g},{p[1]:g}]"
    if t == "normal":
        return f"normal({p[0]:g},{p[1]:g})"
    return f"{t}({raw})"


def prior_width(row):
    if (row.get("prior_type") or "") != "uniform" or not row.get("prior_parameters"):
        return None
    p = [float(v) for v in str(row["prior_parameters"]).split(",")]
    return p[1] - p[0]


def main():
    if not ARCHIVE.exists():
        sys.exit(f"no such archive: {ARCHIVE}")

    rows = [r for r in (collect(f) for f in sorted(ARCHIVE.iterdir()) if f.is_dir()) if r]
    if not rows:
        sys.exit(f"no summary.json found under {ARCHIVE}")

    is_eref = len({r["e_ref"] for r in rows}) > 1
    is_noise = (not is_eref) and len({r["sigma_assumed"] for r in rows}) > 1
    is_prior = (not is_noise) and (not is_eref) and len(
        {r["prior_parameters"] for r in rows}) > 1
    if is_eref:
        rows.sort(key=lambda r: r["e_ref"] or 0)
    elif is_noise:
        rows.sort(key=lambda r: (r["sigma_assumed"] is None, r["sigma_assumed"] or 0))
    elif is_prior:
        rows.sort(key=lambda r: (prior_width(r) is None, prior_width(r) or 0, r["run"]))
    else:
        rows.sort(key=lambda r: (r["load_modulus"] is None,
                                 r["load_modulus"] if r["load_modulus"] else 0, r["run"]))

    out = ARCHIVE / "results.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    def fmt(value, spec):
        return "" if value is None else format(value, spec)

    sigmas = {r["sigma_assumed"] for r in rows if r["sigma_assumed"] is not None}
    if is_noise:
        print(f"sigma swept over {len(sigmas)} values; true noise in the data is 2%")
    elif len(sigmas) == 1:
        s = sigmas.pop()
        pct = 2.0 * s / SIGMA_2PCT
        tag = "matched 2%" if abs(pct - 2.0) < 0.01 else f"{pct:.2f}%  <-- NOT 2%"
        print(f"sigma_assumed = {s:.6e}   ({tag})")
    elif len(sigmas) > 1:
        print("sigma_assumed VARIES across runs")

    show_scaled = (not is_prior) and (not is_noise) and (not is_eref) and any(
        r["alpha_scaled"] is not None for r in rows)

    def noise_text(r):
        s = r["sigma_assumed"]
        return f"{2.0 * s / SIGMA_2PCT:.1f}%" if s else r["label"]

    if is_eref:
        labels = [f"{r['e_ref'] / 1e9:.1f} GPa" for r in rows]
    elif is_noise:
        labels = [noise_text(r) for r in rows]
    elif is_prior:
        labels = [prior_text(r) for r in rows]
    else:
        labels = [r["label"] for r in rows]
    width = max(len(s) for s in labels) + 2

    header = (f"{'run':<{width}}"
              + (f"{'sigma':>13}{'x2%':>7}" if is_noise else "")
              + (f"{'width':>7}" if is_prior else "")
              + f"{'alpha':>9}{'std':>9}{'95% CI':>19}"
              + ("" if is_eref else f"{'bias%':>8}")
              + (f"{'scaled':>9}" if show_scaled else "")
              + (f"{'E [GPa]':>10}" if is_eref else "")
              + f"{'damage':>9}"
              + (f"{'SD/x':>9}" if is_noise else "")
              + f"{'lvls':>6}{'solves':>8}{'logcE':>9}"
              + (f"{'-ln(w)':>9}{'resid':>8}" if is_prior else "")
              + f"{'time_s':>8}")
    print(header)
    print("-" * len(header))

    base = None
    for label, r in zip(labels, rows):
        ci = f"[{r['alpha_p2.5']:.4f},{r['alpha_p97.5']:.4f}]"
        w = prior_width(r)
        occam = -math.log(w) if w else None
        if is_prior and occam is not None and base is None:
            base = r["logcE"] - occam
        resid = (r["logcE"] - occam - base) if (occam is not None and base is not None) else None

        ratio = (r["sigma_assumed"] / SIGMA_2PCT) if r["sigma_assumed"] else None
        e_rec = r["alpha_mean"] * r["e_ref"] if r["e_ref"] else None
        damage = (e_rec / E_UNDAMAGED) if e_rec else None
        print(f"{label:<{width}}"
              + ((f"{r['sigma_assumed']:>13.6e}{ratio:>7.2f}") if is_noise else "")
              + ((fmt(w, '>7.2f') if w else f"{'--':>7}") if is_prior else "")
              + f"{r['alpha_mean']:>9.4f}{r['alpha_std']:>9.4f}{ci:>19}"
              + ("" if is_eref
                 else f"{100 * (r['alpha_mean'] - ALPHA_TRUE):>8.2f}")
              + (fmt(r["alpha_scaled"], ">9.4f") if show_scaled else "")
              + (f"{e_rec / 1e9:>10.2f}" if (is_eref and e_rec) else "")
              + (f"{damage:>9.4f}" if damage else f"{'--':>9}")
              + ((f"{r['alpha_std'] / ratio:>9.4f}" if ratio else f"{'--':>9}")
                 if is_noise else "")
              + f"{r['n_levels']:>6d}{r['n_forward_solves']:>8d}{r['logcE']:>9.3f}"
              + ((fmt(occam, '>9.3f') if occam is not None else f"{'--':>9}")
                 + (fmt(resid, '>8.3f') if resid is not None else f"{'--':>8}") if is_prior else "")
              + fmt(r["wall_time_s"], ">8.0f"))

    if is_eref:
        e = [r["alpha_mean"] * r["e_ref"] / 1e9 for r in rows]
        m = sum(e) / len(e)
        print(f"\nrecovered E: mean {m:.2f} GPa, spread "
              f"{100 * (max(e) - min(e)) / m:.2f}%")
        d = [x / E_UNDAMAGED * 1e9 for x in e]
        print(f"damage ratio E/{E_UNDAMAGED / 1e9:.1f}: {sum(d) / len(d):.4f} "
              f"(1.0 = undamaged); invariant to the guess, unlike alpha")
    elif is_noise:
        a = [r["alpha_mean"] for r in rows]
        spread = 100 * (max(a) - min(a)) / (sum(a) / len(a))
        norm = [r["alpha_std"] * SIGMA_2PCT / r["sigma_assumed"]
                for r in rows if r["sigma_assumed"]]
        print(f"\nalpha spread across sigma: {spread:.3f}%  "
              f"(sigma should not move the mode)")
        print(f"SD/x spread: {100 * (max(norm) - min(norm)) / (sum(norm) / len(norm)):.2f}%  "
              f"(flat => posterior width scales linearly with sigma)")
    elif is_prior:
        a = [r["alpha_mean"] for r in rows]
        spread = 100 * (max(a) - min(a)) / (sum(a) / len(a))
        solves = [r["n_forward_solves"] for r in rows]
        print(f"\nalpha spread across priors: {spread:.3f}%")
        print(f"cost: {min(solves)} -> {max(solves)} forward solves")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
