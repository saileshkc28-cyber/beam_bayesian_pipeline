r"""Builds one table from any sweep archive. Read-only; safe to re-run.

    python collect_sweep.py                          # defaults to Load
    python collect_sweep.py D:\...\Analysis\Prior

Writes results.csv next to the run folders and prints the table.
"""
import csv
import json
import sys
from pathlib import Path

ARCHIVE = Path(sys.argv[1] if len(sys.argv) > 1
               else r"D:\KratosProjects\MCMC\Analysis\Load")

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
    return {
        "run": folder.name,
        "label": info.get("label", folder.name.replace("output_", "")),
        "load_modulus": load,
        "e_ref": info.get("e_ref", zone.get("E_ref_Pa")),
        "sigma_assumed": info.get("sigma_assumed"),
        "prior_type": info.get("prior_type"),
        "prior_parameters": ",".join(str(p) for p in params) if params else None,
        "alpha_mean": zone["alpha_mean"],
        "alpha_std": zone["alpha_std"],
        "alpha_p2.5": zone["alpha_p2.5"],
        "alpha_p97.5": zone["alpha_p97.5"],
        # u ~ F/alpha, so alpha should track the assumed load; flat if that holds
        "alpha_scaled": (zone["alpha_mean"] * 1000.0 / load) if load else None,
        # invariant under reparameterisation: alpha_mean * E_ref should be constant
        "E_mean_Pa": zone["E_mean_Pa"],
        "E_std_Pa": zone["E_std_Pa"],
        "n_levels": len(summary["tempering_q"]),
        "n_forward_solves": summary["n_forward_solves"],
        "wall_time_s": info.get("wall_time_s"),
        "logcE": summary["logcE"],
    }


def main():
    if not ARCHIVE.exists():
        sys.exit(f"no such archive: {ARCHIVE}")

    rows = [r for r in (collect(f) for f in sorted(ARCHIVE.iterdir()) if f.is_dir()) if r]
    if not rows:
        sys.exit(f"no summary.json found under {ARCHIVE}")
    rows.sort(key=lambda r: (r["load_modulus"] is None,
                             r["load_modulus"] if r["load_modulus"] else 0, r["run"]))

    out = ARCHIVE / "results.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    def fmt(value, spec):
        return "" if value is None else format(value, spec)

    show_scaled = any(r["alpha_scaled"] is not None for r in rows)
    show_e = len({r["e_ref"] for r in rows if r["e_ref"]}) > 1
    width = max(len(r["label"]) for r in rows) + 2
    header = (f"{'run':<{width}}{'alpha':>9}{'std':>9}{'95% CI':>19}"
              + (f"{'scaled':>9}" if show_scaled else "")
              + (f"{'E_mean/GPa':>12}" if show_e else "")
              + f"{'lvls':>6}{'solves':>8}{'logcE':>9}{'time_s':>8}")
    print(header)
    print("-" * len(header))
    for r in rows:
        ci = f"[{r['alpha_p2.5']:.4f},{r['alpha_p97.5']:.4f}]"
        print(f"{r['label']:<{width}}{r['alpha_mean']:>9.4f}{r['alpha_std']:>9.4f}{ci:>19}"
              + (fmt(r["alpha_scaled"], ">9.4f") if show_scaled else "")
              + (f"{r['E_mean_Pa'] / 1e9:>12.3f}" if show_e else "")
              + f"{r['n_levels']:>6d}{r['n_forward_solves']:>8d}"
              f"{r['logcE']:>9.3f}{fmt(r['wall_time_s'], '>8.0f')}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
