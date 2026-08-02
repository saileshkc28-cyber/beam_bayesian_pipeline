"""MainBayesian.py — JSON-configured Bayesian inference of Young's modulus.

Usage:
    python MainBayesian.py [BayesianParameters.json] [--sampler metropolis|smc_acs]
                           [--forward kratos|toy]

Everything (parameters, priors, noise, sampler, forward model) comes from the JSON;
the CLI flags are conveniences that override single JSON fields for quick switching.

Sampler contracts (do not mix them up):
    plain Metropolis  <- log_likelihood + log_prior
    ERA SMC_aCS       <- log_likelihood ONLY (the ERADist/ERANataf object holds the
                         prior; its pCN proposal preserves it, so adding the prior
                         here would double-count it)
"""
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from likelihood import Likelihood, Prior, _key
from field_stats import FieldStats
from vtk_writer import write_stats_vtk
from metropolis import run_metropolis, write_table
from closed_form import closed_form_in_s, quadrature_posterior


# ------------------------------------------------------------------ setup
def build_forward_model(cfg):
    kind = cfg["forward_model"]["type"]
    if kind == "kratos":
        from kratos_forward_model import KratosForwardModel
        fm_cfg = cfg["forward_model"]["kratos"]
        return KratosForwardModel(
            fm_cfg["project_parameters_file"],
            cfg["parameters"],
            cfg["likelihood"]["sensor_data_file"],
            self_check=fm_cfg.get("self_check_on_init", True),
        )
    if kind == "toy":
        from toy_forward_model import ToyForwardModel
        return ToyForwardModel(cfg["forward_model"]["toy"]["metadata_file"])
    raise ValueError(f"unknown forward model type '{kind}'")


def field_hook(likelihood, fm):
    def hook(theta):
        return likelihood.field(theta), fm.element_youngs_modulus(theta)
    return hook


# ------------------------------------------------------------------ samplers
def sample_metropolis(cfg, likelihood, prior, fm, out):
    settings = cfg["sampler_settings"]["metropolis"]
    rng = np.random.default_rng(cfg["sampler_settings"].get("random_seed"))
    t0 = time.perf_counter()
    chain, rows, acc, stats = run_metropolis(
        likelihood, prior, settings, rng, field_hook(likelihood, fm))
    wall = time.perf_counter() - t0

    preview = write_table(rows, out / Path(settings["table_file"]).name,
                          settings.get("table_preview_rows", 12))
    print("\n" + preview + "\n")

    if cfg["output"].get("write_vtk", True):
        write_stats_vtk(out / "metropolis_burnin.vtk", fm, *stats["burn"],
                        title="Metropolis burn-in segment")
        write_stats_vtk(out / "metropolis_converged.vtk", fm, *stats["conv"],
                        title="Metropolis converged segment")
        write_stats_vtk(out / "posterior.vtk", fm, *stats["conv"],
                        title="posterior statistics (Metropolis)")

    np.savez(out / "metropolis_chain.npz", chain=chain)
    return {"sampler": "metropolis", "chain": chain, "acceptance_rate": acc,
            "wall_time_s": wall, "logcE": None, "levels": None}


def sample_smc(cfg, likelihood, prior, fm, out):
    sys.path.insert(0, str(Path(__file__).parent / "era"))
    from era.SMC_aCS import SMC_aCS
    s = cfg["sampler_settings"]["smc_acs"]
    np.random.seed(cfg["sampler_settings"].get("random_seed"))   # ERA code uses global state

    t0 = time.perf_counter()
    samplesU, samplesX, q, logcE = SMC_aCS(
        int(s["n_particles"]), s["p"], likelihood.log_likelihood,   # likelihood ONLY
        prior.era_object(), int(s["burn"]), float(s["target_cov"]))
    wall = time.perf_counter() - t0

    levels = [np.atleast_2d(np.asarray(x, float)) for x in samplesX]
    levels = [(x.T if x.shape[0] == prior.dim else x) for x in levels]   # (N, dim)

    # per-level VTK from the evaluation cache (SMC_aCS untouched)
    if cfg["output"].get("write_vtk", True) and fm.mesh_nodes is not None:
        for lev, thetas in enumerate(levels):
            disp, E = FieldStats(), FieldStats()
            for th in thetas:
                cached = likelihood.cache.get(_key(th))
                if cached is None or cached[1] is None:
                    continue
                disp.update(cached[1])
                E.update(fm.element_youngs_modulus(th))
            write_stats_vtk(out / f"smc_level_{lev:02d}.vtk", fm, disp, E,
                            title=f"SMC_aCS tempering level {lev}, q = {q[lev]:.4f}")
        # final posterior
        disp, E = FieldStats(), FieldStats()
        for th in levels[-1]:
            cached = likelihood.cache.get(_key(th))
            if cached is not None and cached[1] is not None:
                disp.update(cached[1])
                E.update(fm.element_youngs_modulus(th))
        write_stats_vtk(out / "posterior.vtk", fm, disp, E,
                        title="posterior statistics (SMC_aCS, q = 1)")

    chain = levels[-1]
    np.savez(out / "smc_levels.npz", q=np.asarray(q), logcE=logcE,
             **{f"level_{i:02d}": lv for i, lv in enumerate(levels)})
    return {"sampler": "smc_acs", "chain": chain, "acceptance_rate": None,
            "wall_time_s": wall, "logcE": float(logcE),
            "levels": [lv for lv in levels], "q": np.asarray(q)}


# ------------------------------------------------------------------ outputs
def make_plots(cfg, result, likelihood, prior, out):
    chain = result["chain"]
    alpha = np.atleast_2d(chain)[:, 0] if chain.ndim > 1 else chain

    # exact references (single-zone only)
    grid = mean_q = std_q = None
    if prior.dim == 1:
        pr = cfg["parameters"][0]["prior"]["parameters"]
        grid, pdf, mean_q, std_q = quadrature_posterior(likelihood, prior,
                                                        lo=pr[0], hi=pr[1])

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(alpha, bins=cfg["output"].get("posterior_bins", 40), density=True,
            color="#4878a8", alpha=0.85, edgecolor="white",
            label=f"{result['sampler']} samples (n={len(alpha)})")
    if grid is not None:
        ax.plot(grid, pdf, "k-", lw=1.6, label="exact posterior (quadrature)")
    ax.axvline(0.7, color="crimson", ls="--", lw=1.4, label=r"$\alpha_{true}=0.7$")
    ax.set_xlabel(r"$\alpha = E/E_{ref}$")
    ax.set_ylabel("posterior density")
    lo = min(alpha.min(), 0.62); hi = max(alpha.max(), 0.78)
    ax.set_xlim(lo - 0.02, hi + 0.02)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"posterior_{result['sampler']}.png", dpi=150)
    plt.close(fig)

    if result["sampler"] == "metropolis":
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(alpha, lw=0.6, color="#4878a8")
        ax.axhline(0.7, color="crimson", ls="--", lw=1)
        ax.set_xlabel("step (post burn-in)"); ax.set_ylabel(r"$\alpha$")
        fig.tight_layout(); fig.savefig(out / "trace_metropolis.png", dpi=150)
        plt.close(fig)

    return mean_q, std_q


def summarise(cfg, result, likelihood, prior, fm, out):
    alpha = np.atleast_2d(result["chain"])
    alpha = alpha[:, 0] if alpha.ndim > 1 else alpha
    mean_q, std_q = make_plots(cfg, result, likelihood, prior, out)

    u_ref = likelihood.forward(np.ones(prior.dim)) * 1.0
    s_hat, s_std = closed_form_in_s(likelihood.u_hat, u_ref, likelihood.sigma)
    s_samples = 1.0 / alpha

    summary = {
        "sampler": result["sampler"],
        "n_forward_solves": fm.n_solves,
        "mean_solve_time_ms": 1e3 * fm.mean_solve_time(),
        "wall_time_s": result["wall_time_s"],
        "acceptance_rate": result["acceptance_rate"],
        "logcE": result["logcE"],
        "tempering_q": (result.get("q").tolist() if result.get("q") is not None else None),
        "posterior_alpha_mean": float(alpha.mean()),
        "posterior_alpha_std": float(alpha.std(ddof=1)),
        "exact_alpha_mean_quadrature": mean_q,
        "exact_alpha_std_quadrature": std_q,
        "closed_form_s_mean": s_hat,
        "closed_form_s_std": s_std,
        "sampled_s_mean": float(s_samples.mean()),
        "sampled_s_std": float(s_samples.std(ddof=1)),
        "cache_size": len(likelihood.cache),
    }
    with open(out / f"summary_{result['sampler']}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"--- {result['sampler']} ---")
    print(f"posterior alpha: mean {alpha.mean():.4f}  std {alpha.std(ddof=1):.4f}")
    if mean_q is not None:
        print(f"exact (quadrature): mean {mean_q:.4f}  std {std_q:.4f}")
    print(f"closed form in s: {s_hat:.4f} +/- {s_std:.4f}   "
          f"sampled s: {s_samples.mean():.4f} +/- {s_samples.std(ddof=1):.4f}")
    if result["logcE"] is not None:
        print(f"log-evidence logcE = {result['logcE']:.3f}")
    if result["acceptance_rate"] is not None:
        print(f"acceptance rate = {result['acceptance_rate']:.3f}")
    print(f"forward solves: {fm.n_solves}  "
          f"({1e3 * fm.mean_solve_time():.2f} ms each, cache size {len(likelihood.cache)})")
    return summary


# ------------------------------------------------------------------ main
def main(argv):
    cfg_file = argv[1] if len(argv) > 1 and not argv[1].startswith("--") else "BayesianParameters.json"
    cfg = json.load(open(cfg_file))
    for flag, path in (("--sampler", ("sampler_settings", "sampler_type")),
                       ("--forward", ("forward_model", "type"))):
        if flag in argv:
            cfg[path[0]][path[1]] = argv[argv.index(flag) + 1]

    out = Path(cfg["output"]["output_path"])
    out.mkdir(parents=True, exist_ok=True)

    fm = build_forward_model(cfg)
    prior = Prior(cfg["parameters"])
    likelihood = Likelihood(fm, cfg["likelihood"]["measured_data_file"],
                            cfg["likelihood"]["noise_model"]["sigma"])

    sampler = cfg["sampler_settings"]["sampler_type"]
    print(f"forward model: {cfg['forward_model']['type']}   sampler: {sampler}\n")
    if sampler == "metropolis":
        result = sample_metropolis(cfg, likelihood, prior, fm, out)
    elif sampler == "smc_acs":
        result = sample_smc(cfg, likelihood, prior, fm, out)
    else:
        raise ValueError(f"unknown sampler_type '{sampler}'")

    return summarise(cfg, result, likelihood, prior, fm, out)


if __name__ == "__main__":
    main(sys.argv)
