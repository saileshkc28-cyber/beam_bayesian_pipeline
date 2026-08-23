import numpy as np
import pints


def run_chains(log_target, initial_states, proposal_std,
               burn_in, draws, random_seed):
    """Drive PINTS MetropolisRandomWalkMCMC with an ask/tell loop.

    log_target is called with ONE parameter vector at a time, so a stateful
    forward model is never entered concurrently.

    NOTE: PINTS treats sigma0 as a COVARIANCE diagonal, so the proposal
    standard deviations are squared here.
    """
    initial_states = np.atleast_2d(np.asarray(initial_states, dtype=float))
    n_chains, dim = initial_states.shape
    variances = np.asarray(proposal_std, dtype=float) ** 2

    np.random.seed(random_seed)          # PINTS uses the global RNG

    samplers = [pints.MetropolisRandomWalkMCMC(x0=x, sigma0=variances)
                for x in initial_states]

    chains = np.empty((n_chains, draws, dim))
    log_posterior = np.empty((n_chains, draws))
    warm_accept = np.zeros(n_chains)
    post_accept = np.zeros(n_chains)
    n_evaluations = 0

    total = burn_in + draws
    for iteration in range(total + 1):       # +1: first ask returns x0
        for c, sampler in enumerate(samplers):
            theta = sampler.ask()
            value = log_target(theta)
            n_evaluations += 1
            current, current_log_pdf, accepted = sampler.tell(value)

            if iteration == 0:
                continue                      # x0 acceptance is unconditional
            if iteration <= burn_in:
                warm_accept[c] += accepted
            else:
                post_accept[c] += accepted
                stored = iteration - burn_in - 1
                chains[c, stored] = current
                log_posterior[c, stored] = current_log_pdf

    return {
        "chains": chains,
        "log_posterior": log_posterior,
        "acceptance_rates": post_accept / draws,
        "warmup_acceptance_rates": warm_accept / burn_in,
        "n_target_evaluations": n_evaluations,
    }
