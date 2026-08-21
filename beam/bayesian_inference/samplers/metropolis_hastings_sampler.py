import numpy as np
import KratosMultiphysics as Kratos

from mcmc.metropolis_hastings import (
    RandomWalkMetropolisHastings
)


def Factory(likelihood, prior, settings):
    return MetropolisHastingsSampler(
        likelihood,
        prior,
        settings
    )


class MetropolisHastingsSampler:
    """Kratos adapter for plain random-walk Metropolis-Hastings."""

    @staticmethod
    def GetDefaultParameters():
        return Kratos.Parameters("""{
            "python_module" : "metropolis_hastings_sampler",
            "Parameters"    : {
                "n_chains"       : 4,
                "burn_in"        : 1000,
                "draws_per_chain": 2500,
                "proposal_std"   : [0.04],
                "initial_states" : [
                    [0.35],
                    [0.75],
                    [1.25],
                    [1.75]
                ],
                "random_seed"    : 20260822
            }
        }""")

    def __init__(
        self,
        likelihood,
        prior,
        settings
    ):
        settings.ValidateAndAssignDefaults(
            self.GetDefaultParameters()["Parameters"]
        )

        self.likelihood = likelihood
        self.prior = prior
        self.settings = settings

        self.result = None
        self.chains = None
        self.posterior = None
        self.acceptance_rates = None
        self.warmup_acceptance_rates = None

    def _LogPrior(self, theta):
        """Evaluate independent prior marginals."""
        theta = np.atleast_1d(
            np.asarray(theta, dtype=float)
        )

        if theta.size != self.prior.dim:
            raise ValueError(
                f"expected {self.prior.dim} parameters, "
                f"received {theta.size}"
            )

        log_prior = 0.0

        for value, marginal in zip(
            theta,
            self.prior.marginals
        ):
            density = float(
                np.asarray(
                    marginal.pdf(value),
                    dtype=float
                ).reshape(-1)[0]
            )

            if (
                not np.isfinite(density)
                or density <= 0.0
            ):
                return -np.inf

            log_prior += np.log(density)

        return float(log_prior)

    def _LogPosterior(self, theta):
        """Return log-prior plus Kratos-based log-likelihood."""
        log_prior = self._LogPrior(theta)

        if not np.isfinite(log_prior):
            return -np.inf

        log_likelihood = float(
            self.likelihood(theta)
        )

        if not np.isfinite(log_likelihood):
            return -np.inf

        return log_prior + log_likelihood

    def _ReadInitialStates(self):
        initial_parameters = self.settings[
            "initial_states"
        ]

        initial_states = np.array(
            [
                list(
                    initial_parameters[
                        index
                    ].GetVector()
                )
                for index in range(
                    initial_parameters.size()
                )
            ],
            dtype=float
        )

        n_chains = self.settings[
            "n_chains"
        ].GetInt()

        if initial_states.shape != (
            n_chains,
            self.prior.dim
        ):
            raise ValueError(
                "initial_states must have shape "
                f"({n_chains}, {self.prior.dim}), "
                f"received {initial_states.shape}"
            )

        return initial_states

    def Run(self):
        initial_states = self._ReadInitialStates()

        sampler = RandomWalkMetropolisHastings(
            log_target=self._LogPosterior,
            proposal_std=list(
                self.settings[
                    "proposal_std"
                ].GetVector()
            ),
            burn_in=self.settings[
                "burn_in"
            ].GetInt(),
            draws=self.settings[
                "draws_per_chain"
            ].GetInt(),
            random_seed=self.settings[
                "random_seed"
            ].GetInt()
        )

        self.result = sampler.Run(
            initial_states=initial_states
        )

        self.chains = self.result.chains
        self.posterior = self.result.posterior
        self.acceptance_rates = (
            self.result.acceptance_rates
        )
        self.warmup_acceptance_rates = (
            self.result.warmup_acceptance_rates
        )

        return self.posterior

    def AlgorithmName(self):
        return "RandomWalkMetropolisHastings"

    def Posterior(self):
        if self.posterior is None:
            raise RuntimeError(
                "Run() must be called before Posterior()"
            )

        return self.posterior

    def Evidence(self):
        """Plain Metropolis-Hastings does not estimate evidence."""
        return None

    def Diagnostics(self):
        if self.result is None:
            raise RuntimeError(
                "Run() must be called before Diagnostics()"
            )

        return {
            "acceptance_rates": (
                self.acceptance_rates.tolist()
            ),
            "warmup_acceptance_rates": (
                self.warmup_acceptance_rates.tolist()
            ),
            "n_target_evaluations": (
                self.result.n_target_evaluations
            )
        }