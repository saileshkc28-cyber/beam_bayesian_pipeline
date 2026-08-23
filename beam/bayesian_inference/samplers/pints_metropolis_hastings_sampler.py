import numpy as np
import KratosMultiphysics as Kratos

from mcmc.pints_driver import run_chains


class PintsResult:
    """Attribute-style view of the driver output, matching the result object
    produced by RandomWalkMetropolisHastings so the existing output processes
    work unchanged."""

    def __init__(self, data):
        self.chains = data["chains"]
        self.log_posterior = data["log_posterior"]
        self.acceptance_rates = data["acceptance_rates"]
        self.warmup_acceptance_rates = data["warmup_acceptance_rates"]
        self.n_target_evaluations = data["n_target_evaluations"]
        self.posterior = self.chains.reshape(-1, self.chains.shape[2])


def Factory(likelihood, prior, settings):
    return PintsMetropolisHastingsSampler(likelihood, prior, settings)


class PintsMetropolisHastingsSampler:
    """Kratos adapter for the PINTS random-walk Metropolis sampler.

    Same algorithm and same interface as MetropolisHastingsSampler; only the
    accept/reject kernel is external (pints.MetropolisRandomWalkMCMC).
    """

    @staticmethod
    def GetDefaultParameters():
        return Kratos.Parameters("""{
            "python_module" : "pints_metropolis_hastings_sampler",
            "Parameters"    : {
                "n_chains"        : 4,
                "burn_in"         : 500,
                "draws_per_chain" : 1000,
                "proposal_std"    : [0.06],
                "initial_states"  : [
                    [0.35],
                    [0.75],
                    [1.25],
                    [1.75]
                ],
                "random_seed"     : 20260822
            }
        }""")

    def __init__(self, likelihood, prior, settings):
        settings.ValidateAndAssignDefaults(
            self.GetDefaultParameters()["Parameters"])

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
        theta = np.atleast_1d(np.asarray(theta, dtype=float))

        if theta.size != self.prior.dim:
            raise ValueError(
                f"expected {self.prior.dim} parameters, received {theta.size}")

        log_prior = 0.0
        for value, marginal in zip(theta, self.prior.marginals):
            density = float(
                np.asarray(marginal.pdf(value), dtype=float).reshape(-1)[0])
            if not np.isfinite(density) or density <= 0.0:
                return -np.inf
            log_prior += np.log(density)

        return float(log_prior)

    def _LogPosterior(self, theta):
        """Log-prior plus Kratos-based log-likelihood.

        Points outside the prior support return -inf without a Kratos solve.
        """
        log_prior = self._LogPrior(theta)
        if not np.isfinite(log_prior):
            return -np.inf

        log_likelihood = float(self.likelihood(theta))
        if not np.isfinite(log_likelihood):
            return -np.inf

        return log_prior + log_likelihood

    def _ReadInitialStates(self):
        entries = self.settings["initial_states"]
        initial_states = np.array(
            [list(entries[i].GetVector()) for i in range(entries.size())],
            dtype=float)

        n_chains = self.settings["n_chains"].GetInt()
        if initial_states.shape != (n_chains, self.prior.dim):
            raise ValueError(
                "initial_states must have shape "
                f"({n_chains}, {self.prior.dim}), "
                f"received {initial_states.shape}")

        return initial_states

    def Run(self):
        initial_states = self._ReadInitialStates()

        proposal_std = list(self.settings["proposal_std"].GetVector())
        if len(proposal_std) != self.prior.dim:
            raise ValueError(
                f"proposal_std must have {self.prior.dim} entries, "
                f"received {len(proposal_std)}")

        self.result = PintsResult(run_chains(
            log_target=self._LogPosterior,
            initial_states=initial_states,
            proposal_std=proposal_std,
            burn_in=self.settings["burn_in"].GetInt(),
            draws=self.settings["draws_per_chain"].GetInt(),
            random_seed=self.settings["random_seed"].GetInt()))

        self.chains = self.result.chains
        self.acceptance_rates = self.result.acceptance_rates
        self.warmup_acceptance_rates = self.result.warmup_acceptance_rates
        self.posterior = self.result.posterior

        return self.posterior

    def AlgorithmName(self):
        return "PINTS.MetropolisRandomWalkMCMC"

    def Posterior(self):
        if self.posterior is None:
            raise RuntimeError("Run() must be called before Posterior()")
        return self.posterior

    def Evidence(self):
        """Plain Metropolis-Hastings does not estimate evidence."""
        return None

    def Diagnostics(self):
        if self.result is None:
            raise RuntimeError("Run() must be called before Diagnostics()")

        return {
            "acceptance_rates": self.acceptance_rates.tolist(),
            "warmup_acceptance_rates": self.warmup_acceptance_rates.tolist(),
            "n_target_evaluations": self.result.n_target_evaluations,
        }
