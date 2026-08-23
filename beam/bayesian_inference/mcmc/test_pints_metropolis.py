"""Standalone check of the PINTS driver against a known normal target.

Mirrors mcmc/test_metropolis_hastings.py so the two are comparable.
No Kratos involved.
"""
import numpy as np

from mcmc.pints_driver import run_chains

MEAN, STD = 1.25, 0.4


def log_target(theta):
    return float(-0.5 * ((np.atleast_1d(theta)[0] - MEAN) / STD) ** 2)


def main():
    result = run_chains(
        log_target=log_target,
        initial_states=[[0.35], [0.75], [1.25], [1.75]],
        proposal_std=[0.5],
        burn_in=2000,
        draws=10000,
        random_seed=20260822,
    )

    chains = result["chains"]
    pooled = chains.reshape(-1)

    print("chain shape          :", chains.shape)
    print("acceptance rates     :", np.round(result["acceptance_rates"], 4))
    print("warmup acceptance    :", np.round(result["warmup_acceptance_rates"], 4))
    print("target evaluations   :", result["n_target_evaluations"])
    print("estimated mean       : %.6f  (expected %.4f)" % (pooled.mean(), MEAN))
    print("estimated std        : %.6f  (expected %.4f)" % (pooled.std(ddof=1), STD))


if __name__ == "__main__":
    main()
