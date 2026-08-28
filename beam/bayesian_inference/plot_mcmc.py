"""Post-process saved Metropolis-Hastings chains.

This script does not run Kratos or MCMC. It only reads the saved
MCMC results and creates diagnostic figures.

Two stacked panels per parameter:
  top    - combined posterior histogram
  bottom - running mean stability per chain
"""

import json
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DATA_FILE = "output_mcmc/mcmc_chains.npz"
TRUTH_FILE = "../damaged_system/StructuralMaterials.json"
OUTPUT_DIRECTORY = "output_mcmc"
ALGORITHM_LABEL = "Random-walk Metropolis-Hastings (own implementation)"


def split_rhat(chains):
    """Calculate the conventional split-R-hat diagnostic."""
    chains = np.asarray(chains, dtype=float)
    number_of_chains, number_of_draws = chains.shape
    half_length = number_of_draws // 2

    if number_of_chains < 2 or half_length < 2:
        return np.nan

    split_chains = np.concatenate(
        (chains[:, :half_length], chains[:, -half_length:]), axis=0)

    draws_per_split_chain = split_chains.shape[1]
    chain_means = np.mean(split_chains, axis=1)

    within_chain_variance = float(
        np.mean(np.var(split_chains, axis=1, ddof=1)))
    between_chain_variance = float(
        draws_per_split_chain * np.var(chain_means, ddof=1))

    if within_chain_variance == 0.0:
        return np.nan

    estimated_variance = (
        (draws_per_split_chain - 1) / draws_per_split_chain
        * within_chain_variance
        + between_chain_variance / draws_per_split_chain)

    return float(np.sqrt(estimated_variance / within_chain_variance))


def read_truth(truth_file, property_id, reference_value):
    """Read the true alpha value when a truth file is available."""
    if not os.path.exists(truth_file):
        return None

    with open(truth_file, "r") as input_file:
        material_data = json.load(input_file)

    for block in material_data["properties"]:
        if block["properties_id"] == property_id:
            youngs_modulus = block["Material"]["Variables"]["YOUNG_MODULUS"]
            return float(youngs_modulus) / reference_value

    return None


def create_zone_figure(chains, acceptance_rates, reference_value,
                       truth, zone_index):
    """Create the combined-posterior and running-mean panels."""
    number_of_chains, number_of_draws = chains.shape
    draw_numbers = np.arange(1, number_of_draws + 1)
    colors = plt.cm.tab10(np.linspace(0.0, 0.75, number_of_chains))

    combined_samples = chains.reshape(-1)
    combined_mean = float(np.mean(combined_samples))
    combined_std = float(np.std(combined_samples, ddof=1))
    rhat = split_rhat(chains)

    figure = plt.figure(figsize=(11, 13))
    grid = figure.add_gridspec(3, number_of_chains, height_ratios=[1.15, 0.75, 1.0],
                               hspace=0.45, wspace=0.30)

    # ---------------------------------------------- combined posterior
    posterior_axis = figure.add_subplot(grid[0, :])

    posterior_axis.hist(
        combined_samples, bins=35, density=True, color="#4c78a8",
        edgecolor="white", alpha=0.80,
        label=(rf"combined: $\alpha={combined_mean:.4f}"
               rf"\pm{combined_std:.4f}$"))

    posterior_axis.axvline(combined_mean, color="#1a3a5a", linewidth=1.8)
    for bound in (combined_mean - combined_std, combined_mean + combined_std):
        posterior_axis.axvline(bound, color="#1a3a5a",
                               linestyle=":", linewidth=1.0)

    if truth is not None:
        posterior_axis.axvline(
            truth, color="crimson", linestyle="--", linewidth=1.5,
            label=rf"$\alpha_{{true}}={truth:.4f}$")

    posterior_axis.set_xlabel(r"$\alpha=E/E_{ref}$")
    posterior_axis.set_ylabel("Density")
    posterior_axis.set_title("Combined posterior", fontsize=11)
    posterior_axis.grid(True, axis="y", alpha=0.20)
    posterior_axis.legend(frameon=False, fontsize=8)

    secondary_axis = posterior_axis.secondary_xaxis(
        "top",
        functions=(lambda alpha: alpha * reference_value / 1.0e9,
                   lambda youngs_modulus: youngs_modulus * 1.0e9 / reference_value))
    secondary_axis.set_xlabel("Young's modulus [GPa]", fontsize=9)

    # ---------------------------------------------- per-chain posteriors
    shared_bins = np.linspace(combined_samples.min(), combined_samples.max(), 30)

    for chain_index in range(number_of_chains):
        chain_axis = figure.add_subplot(grid[1, chain_index])
        chain_samples = chains[chain_index]
        chain_mean = float(np.mean(chain_samples))

        chain_axis.hist(chain_samples, bins=shared_bins, density=True,
                        color=colors[chain_index], edgecolor="white", alpha=0.85)
        chain_axis.axvline(chain_mean, color="#1a3a5a", linewidth=1.4)

        if truth is not None:
            chain_axis.axvline(truth, color="crimson", linestyle="--", linewidth=1.2)

        chain_axis.set_xlim(shared_bins[0], shared_bins[-1])
        chain_axis.set_title(
            f"chain {chain_index + 1}\n"
            rf"$\alpha$ = {chain_mean:.4f} $\pm$ {np.std(chain_samples, ddof=1):.4f}",
            fontsize=9)
        chain_axis.set_xlabel(r"$\alpha$", fontsize=8)
        chain_axis.tick_params(labelsize=7)
        chain_axis.grid(True, axis="y", alpha=0.20)

        if chain_index == 0:
            chain_axis.set_ylabel("Density", fontsize=8)
        else:
            chain_axis.set_yticklabels([])

    # ---------------------------------------------- running mean
    running_axis = figure.add_subplot(grid[2, :])

    for chain_index in range(number_of_chains):
        running_mean = np.cumsum(chains[chain_index]) / draw_numbers
        running_axis.plot(draw_numbers, running_mean,
                          color=colors[chain_index], linewidth=1.1,
                          label=f"chain {chain_index + 1}")

    running_axis.axhline(combined_mean, color="#1a3a5a", linestyle=":",
                         linewidth=1.5, label="combined mean")

    if truth is not None:
        running_axis.axhline(truth, color="crimson", linestyle="--",
                             linewidth=1.5, label=r"$\alpha_{true}$")

    running_axis.set_xlabel("Stored draw")
    running_axis.set_ylabel(r"Running mean of $\alpha$")
    running_axis.set_title("Running mean stability", fontsize=11)
    running_axis.grid(True, alpha=0.20)
    running_axis.legend(frameon=False, fontsize=8)

    figure.suptitle(
        (f"{ALGORITHM_LABEL} — parameter {zone_index + 1}\n"
         f"split R-hat = {rhat:.4f}, "
         f"mean acceptance = {np.mean(acceptance_rates):.3f}"),
        fontsize=13)

    figure.subplots_adjust(top=0.90, bottom=0.06, left=0.08, right=0.96)

    output_file = os.path.join(
        OUTPUT_DIRECTORY,
        f"mcmc_diagnostics_alpha_{zone_index + 1}.png")

    figure.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close(figure)

    return output_file, combined_mean, combined_std, rhat


def main():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    data = np.load(DATA_FILE)
    chains = np.asarray(data["chains"], dtype=float)
    acceptance_rates = np.asarray(data["acceptance_rates"], dtype=float)
    reference_values = np.asarray(data["E_ref"], dtype=float)
    property_ids = np.asarray(data["property_ids"], dtype=int)

    print(f"{'chain':>7} {'acceptance':>12} {'mean':>12} {'std':>12}")

    for chain_index in range(chains.shape[0]):
        print(f"{chain_index + 1:>7d} "
              f"{acceptance_rates[chain_index]:>12.3f} "
              f"{np.mean(chains[chain_index, :, 0]):>12.5f} "
              f"{np.std(chains[chain_index, :, 0], ddof=1):>12.5f}")

    for zone_index in range(chains.shape[2]):
        truth = read_truth(TRUTH_FILE,
                           int(property_ids[zone_index]),
                           float(reference_values[zone_index]))

        output_file, posterior_mean, posterior_std, rhat = create_zone_figure(
            chains=chains[:, :, zone_index],
            acceptance_rates=acceptance_rates,
            reference_value=float(reference_values[zone_index]),
            truth=truth,
            zone_index=zone_index)

        print()
        print(f"parameter {zone_index + 1}: "
              f"mean = {posterior_mean:.6f}, "
              f"std = {posterior_std:.6f}, "
              f"split R-hat = {rhat:.4f}")
        print(f"wrote {output_file}")


if __name__ == "__main__":
    main()