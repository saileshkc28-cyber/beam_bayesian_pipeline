import numpy as np

from mcmc.metropolis_hastings import (
    RandomWalkMetropolisHastings
)


TRUE_MEAN = 1.25
TRUE_STD = 0.40


def log_target(theta):
    """Log-density proportional to Normal(1.25, 0.40²)."""
    x = float(theta[0])

    return -0.5 * (
        (x - TRUE_MEAN) / TRUE_STD
    ) ** 2


sampler = RandomWalkMetropolisHastings(
    log_target=log_target,
    proposal_std=[0.50],
    burn_in=2_000,
    draws=10_000,
    random_seed=20260822
)

result = sampler.Run(
    initial_states=[
        [-1.00],
        [0.00],
        [2.50],
        [4.00]
    ]
)

samples = result.posterior[:, 0]

estimated_mean = float(
    np.mean(samples)
)

estimated_std = float(
    np.std(samples, ddof=1)
)

print("Chain shape:", result.chains.shape)
print("Acceptance rates:", result.acceptance_rates)
print("Estimated mean:", estimated_mean)
print("Expected mean:", TRUE_MEAN)
print("Estimated std:", estimated_std)
print("Expected std:", TRUE_STD)
print(
    "Target evaluations:",
    result.n_target_evaluations
)

assert result.chains.shape == (4, 10_000, 1)
assert abs(estimated_mean - TRUE_MEAN) < 0.03
assert abs(estimated_std - TRUE_STD) < 0.03
assert np.all(result.acceptance_rates > 0.20)
assert np.all(result.acceptance_rates < 0.80)

print("Metropolis-Hastings test passed.")