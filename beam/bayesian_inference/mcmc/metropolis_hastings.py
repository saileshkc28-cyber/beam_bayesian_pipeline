from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MCMCResult:
    """Results returned by the Metropolis-Hastings sampler."""

    chains: np.ndarray
    log_posterior: np.ndarray
    acceptance_rates: np.ndarray
    warmup_acceptance_rates: np.ndarray
    n_target_evaluations: int

    @property
    def posterior(self):
        """Combine all chains into one posterior sample array."""
        return self.chains.reshape(
            -1,
            self.chains.shape[-1]
        )


class RandomWalkMetropolisHastings:
    """Plain random-walk Metropolis-Hastings sampler.

    A symmetric Gaussian proposal is used:

        candidate = current + Normal(0, proposal_std)

    The proposal-density terms therefore cancel from the
    Metropolis-Hastings acceptance ratio.
    """

    def __init__(
        self,
        log_target,
        proposal_std,
        burn_in,
        draws,
        random_seed=0
    ):
        self.log_target = log_target

        self.proposal_std = np.atleast_1d(
            np.asarray(proposal_std, dtype=float)
        )

        self.burn_in = int(burn_in)
        self.draws = int(draws)
        self.random_seed = int(random_seed)

        if np.any(self.proposal_std <= 0.0):
            raise ValueError(
                "all proposal standard deviations must be positive"
            )

        if self.burn_in < 0:
            raise ValueError(
                "burn_in must be zero or positive"
            )

        if self.draws <= 0:
            raise ValueError(
                "draws must be positive"
            )

    def _EvaluateLogTarget(self, position):
        """Evaluate the target and require one scalar log-density."""
        value = np.asarray(
            self.log_target(position),
            dtype=float
        )

        if value.size != 1:
            raise ValueError(
                "log_target must return one scalar value"
            )

        return float(value.reshape(-1)[0])

    def Run(self, initial_states):
        """Run independent chains sequentially.

        Parameters
        ----------
        initial_states : array-like
            Shape: (number_of_chains, number_of_parameters).

        Returns
        -------
        MCMCResult
            Stored post-warmup chains and diagnostics.
        """
        initial_states = np.asarray(
            initial_states,
            dtype=float
        )

        if initial_states.ndim != 2:
            raise ValueError(
                "initial_states must have shape "
                "(number_of_chains, number_of_parameters)"
            )

        n_chains, dimension = initial_states.shape

        if self.proposal_std.size == 1:
            proposal_std = np.full(
                dimension,
                self.proposal_std[0],
                dtype=float
            )
        elif self.proposal_std.size == dimension:
            proposal_std = self.proposal_std.copy()
        else:
            raise ValueError(
                "proposal_std must contain either one value "
                "or one value per parameter"
            )

        chains = np.empty(
            (n_chains, self.draws, dimension),
            dtype=float
        )

        log_posterior = np.empty(
            (n_chains, self.draws),
            dtype=float
        )

        acceptance_rates = np.zeros(
            n_chains,
            dtype=float
        )

        warmup_acceptance_rates = np.zeros(
            n_chains,
            dtype=float
        )

        seed_sequence = np.random.SeedSequence(
            self.random_seed
        )

        chain_seeds = seed_sequence.spawn(n_chains)

        n_target_evaluations = 0

        for chain_index in range(n_chains):
            rng = np.random.default_rng(
                chain_seeds[chain_index]
            )

            current = initial_states[
                chain_index
            ].copy()

            current_log_target = self._EvaluateLogTarget(
                current
            )

            n_target_evaluations += 1

            if not np.isfinite(current_log_target):
                raise ValueError(
                    f"initial state of chain "
                    f"{chain_index + 1} has a non-finite "
                    "log-target value"
                )

            accepted_warmup = 0
            accepted_sampling = 0

            total_iterations = (
                self.burn_in + self.draws
            )

            for iteration in range(total_iterations):
                candidate = current + rng.normal(
                    loc=0.0,
                    scale=proposal_std,
                    size=dimension
                )

                candidate_log_target = (
                    self._EvaluateLogTarget(candidate)
                )

                n_target_evaluations += 1

                log_acceptance_ratio = (
                    candidate_log_target
                    - current_log_target
                )

                accepted = (
                    np.isfinite(candidate_log_target)
                    and np.log(rng.random())
                    < log_acceptance_ratio
                )

                if accepted:
                    current = candidate
                    current_log_target = (
                        candidate_log_target
                    )

                if iteration < self.burn_in:
                    accepted_warmup += int(accepted)
                    continue

                sample_index = (
                    iteration - self.burn_in
                )

                chains[
                    chain_index,
                    sample_index
                ] = current

                log_posterior[
                    chain_index,
                    sample_index
                ] = current_log_target

                accepted_sampling += int(accepted)

            if self.burn_in > 0:
                warmup_acceptance_rates[
                    chain_index
                ] = (
                    accepted_warmup
                    / self.burn_in
                )
            else:
                warmup_acceptance_rates[
                    chain_index
                ] = np.nan

            acceptance_rates[
                chain_index
            ] = (
                accepted_sampling
                / self.draws
            )

        return MCMCResult(
            chains=chains,
            log_posterior=log_posterior,
            acceptance_rates=acceptance_rates,
            warmup_acceptance_rates=(
                warmup_acceptance_rates
            ),
            n_target_evaluations=(
                n_target_evaluations
            )
        )