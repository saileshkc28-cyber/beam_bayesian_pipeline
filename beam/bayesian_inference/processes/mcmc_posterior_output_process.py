import json
import os

import numpy as np
import KratosMultiphysics as Kratos

from arviz_stats.base import array_stats


def Factory(
    forward_model,
    likelihood,
    settings
):
    return MCMCPosteriorOutputProcess(
        forward_model,
        settings
    )


class MCMCPosteriorOutputProcess:
    """Save MCMC chains, posterior samples and summary statistics."""

    @staticmethod
    def GetDefaultParameters():
        return Kratos.Parameters("""{
            "python_module": "mcmc_posterior_output_process",
            "Parameters": {
                "output_path": "output_mcmc",
                "write_npz": true,
                "write_excel": true
            }
        }""")

    def __init__(
        self,
        forward_model,
        settings
    ):
        settings.ValidateAndAssignDefaults(
            self.GetDefaultParameters()["Parameters"]
        )

        self.model = forward_model

        self.output_path = settings[
            "output_path"
        ].GetString()

        self.write_npz = settings[
            "write_npz"
        ].GetBool()

        self.write_excel = settings[
            "write_excel"
        ].GetBool()

    def ExecuteFinalize(self, sampler):
        if sampler.result is None:
            raise RuntimeError(
                "MCMC sampler has no results"
            )

        os.makedirs(
            self.output_path,
            exist_ok=True
        )

        chains = np.asarray(
            sampler.chains,
            dtype=float
        )

        posterior = np.asarray(
            sampler.Posterior(),
            dtype=float
        )

        log_posterior = np.asarray(
            sampler.result.log_posterior,
            dtype=float
        )

        acceptance_rates = np.asarray(
            sampler.acceptance_rates,
            dtype=float
        )

        warmup_acceptance_rates = np.asarray(
            sampler.warmup_acceptance_rates,
            dtype=float
        )

        reference_values = np.asarray(
            self.model.refs,
            dtype=float
        )

        if chains.shape[2] != reference_values.size:
            raise ValueError(
                "number of MCMC parameters does not match "
                "the number of reference material values"
            )

        youngs_modulus_chains = (
            chains
            * reference_values[
                np.newaxis,
                np.newaxis,
                :
            ]
        )

        youngs_modulus_posterior = (
            posterior
            * reference_values[
                np.newaxis,
                :
            ]
        )

        statistics = self._CalculateStatistics(
            posterior,
            reference_values
        )

        convergence_diagnostics = (
            self._CalculateConvergenceDiagnostics(
                chains
            )
        )

        rank_normalized_rhat = np.array(
            [
                diagnostic[
                    "rank_normalized_rhat"
                ]
                for diagnostic
                in convergence_diagnostics
            ],
            dtype=float
        )

        bulk_ess = np.array(
            [
                diagnostic["bulk_ess"]
                for diagnostic
                in convergence_diagnostics
            ],
            dtype=float
        )

        tail_ess = np.array(
            [
                diagnostic["tail_ess"]
                for diagnostic
                in convergence_diagnostics
            ],
            dtype=float
        )

        if self.write_npz:
            np.savez(
                os.path.join(
                    self.output_path,
                    "mcmc_chains.npz"
                ),
                chains=chains,
                posterior=posterior,
                log_posterior=log_posterior,
                acceptance_rates=acceptance_rates,
                warmup_acceptance_rates=(
                    warmup_acceptance_rates
                ),
                rank_normalized_rhat=(
                    rank_normalized_rhat
                ),
                bulk_ess=bulk_ess,
                tail_ess=tail_ess,
                E_ref=reference_values,
                E_chains=youngs_modulus_chains,
                E_posterior=(
                    youngs_modulus_posterior
                ),
                property_ids=np.array(
                    [
                        property_ids[0]
                        for property_ids
                        in self.model.prop_ids
                    ],
                    dtype=int
                )
            )

        summary = {
            "algorithm": sampler.AlgorithmName(),
            "n_chains": int(chains.shape[0]),
            "draws_per_chain": int(chains.shape[1]),
            "n_posterior_samples": int(
                posterior.shape[0]
            ),
            "burn_in": sampler.settings[
                "burn_in"
            ].GetInt(),
            "proposal_std": list(
                sampler.settings[
                    "proposal_std"
                ].GetVector()
            ),
            "random_seed": sampler.settings[
                "random_seed"
            ].GetInt(),
            "acceptance_rates": (
                acceptance_rates.tolist()
            ),
            "mean_acceptance_rate": float(
                np.mean(acceptance_rates)
            ),
            "warmup_acceptance_rates": (
                warmup_acceptance_rates.tolist()
            ),
            "n_target_evaluations": int(
                sampler.result.n_target_evaluations
            ),
            "n_forward_solves": int(
                self.model.n_solves
            ),
            "convergence_diagnostics": (
                convergence_diagnostics
            ),
            "zones": statistics
        }

        with open(
            os.path.join(
                self.output_path,
                "mcmc_summary.json"
            ),
            "w"
        ) as output_file:
            json.dump(
                summary,
                output_file,
                indent=2
            )

        if self.write_excel:
            self._WriteExcel(
                chains,
                youngs_modulus_chains,
                log_posterior,
                acceptance_rates,
                warmup_acceptance_rates,
                statistics,
                convergence_diagnostics
            )

        for zone in statistics:
            Kratos.Logger.PrintInfo(
                "MCMCPosteriorOutput",
                (
                    "zone %d: "
                    "alpha = %.4f +/- %.4f, "
                    "E = %.4e +/- %.3e Pa"
                )
                % (
                    zone["zone"],
                    zone["alpha_mean"],
                    zone["alpha_std"],
                    zone["E_mean_Pa"],
                    zone["E_std_Pa"]
                )
            )

        for diagnostic in convergence_diagnostics:
            Kratos.Logger.PrintInfo(
                "MCMCPosteriorOutput",
                (
                    "parameter %d: "
                    "rank-normalized R-hat = %.4f, "
                    "bulk ESS = %.1f, "
                    "tail ESS = %.1f"
                )
                % (
                    diagnostic["parameter"],
                    diagnostic[
                        "rank_normalized_rhat"
                    ],
                    diagnostic["bulk_ess"],
                    diagnostic["tail_ess"]
                )
            )

    def _CalculateConvergenceDiagnostics(
        self,
        chains
    ):
        rank_normalized_rhat = np.asarray(
            array_stats.rhat(
                chains,
                method="rank",
                chain_axis=0,
                draw_axis=1
            ),
            dtype=float
        ).reshape(-1)

        bulk_ess = np.asarray(
            array_stats.ess(
                chains,
                method="bulk",
                chain_axis=0,
                draw_axis=1
            ),
            dtype=float
        ).reshape(-1)

        tail_ess = np.asarray(
            array_stats.ess(
                chains,
                method="tail",
                prob=(0.05, 0.95),
                chain_axis=0,
                draw_axis=1
            ),
            dtype=float
        ).reshape(-1)

        if not (
            rank_normalized_rhat.size
            == bulk_ess.size
            == tail_ess.size
            == chains.shape[2]
        ):
            raise RuntimeError(
                "unexpected MCMC diagnostic dimensions"
            )

        total_draws = int(
            chains.shape[0]
            * chains.shape[1]
        )

        diagnostics = []

        for parameter_index in range(
            chains.shape[2]
        ):
            diagnostics.append(
                {
                    "parameter": parameter_index + 1,
                    "rank_normalized_rhat": float(
                        rank_normalized_rhat[
                            parameter_index
                        ]
                    ),
                    "bulk_ess": float(
                        bulk_ess[
                            parameter_index
                        ]
                    ),
                    "tail_ess": float(
                        tail_ess[
                            parameter_index
                        ]
                    ),
                    "relative_bulk_ess": float(
                        bulk_ess[
                            parameter_index
                        ]
                        / total_draws
                    ),
                    "relative_tail_ess": float(
                        tail_ess[
                            parameter_index
                        ]
                        / total_draws
                    )
                }
            )

        return diagnostics

    def _CalculateStatistics(
        self,
        posterior,
        reference_values
    ):
        statistics = []

        for zone_index in range(
            posterior.shape[1]
        ):
            alpha_samples = posterior[
                :,
                zone_index
            ]

            reference_value = reference_values[
                zone_index
            ]

            statistics.append(
                {
                    "zone": zone_index + 1,
                    "alpha_mean": float(
                        np.mean(alpha_samples)
                    ),
                    "alpha_std": float(
                        np.std(
                            alpha_samples,
                            ddof=1
                        )
                    ),
                    "alpha_p2.5": float(
                        np.percentile(
                            alpha_samples,
                            2.5
                        )
                    ),
                    "alpha_p50": float(
                        np.percentile(
                            alpha_samples,
                            50.0
                        )
                    ),
                    "alpha_p97.5": float(
                        np.percentile(
                            alpha_samples,
                            97.5
                        )
                    ),
                    "E_ref_Pa": float(
                        reference_value
                    ),
                    "E_mean_Pa": float(
                        np.mean(alpha_samples)
                        * reference_value
                    ),
                    "E_std_Pa": float(
                        np.std(
                            alpha_samples,
                            ddof=1
                        )
                        * reference_value
                    )
                }
            )

        return statistics

    def _WriteExcel(
        self,
        chains,
        youngs_modulus_chains,
        log_posterior,
        acceptance_rates,
        warmup_acceptance_rates,
        statistics,
        convergence_diagnostics
    ):
        try:
            import pandas as pd
            import openpyxl
        except ImportError:
            Kratos.Logger.PrintWarning(
                "MCMCPosteriorOutput",
                "pandas/openpyxl missing - skipping xlsx"
            )
            return

        excel_path = os.path.join(
            self.output_path,
            "mcmc_posterior.xlsx"
        )

        with pd.ExcelWriter(
            excel_path,
            engine="openpyxl"
        ) as writer:
            for chain_index in range(
                chains.shape[0]
            ):
                columns = {
                    "draw": np.arange(
                        1,
                        chains.shape[1] + 1
                    ),
                    "log_posterior": log_posterior[
                        chain_index
                    ]
                }

                for zone_index in range(
                    chains.shape[2]
                ):
                    columns[
                        f"alpha_{zone_index + 1}"
                    ] = chains[
                        chain_index,
                        :,
                        zone_index
                    ]

                    columns[
                        f"E_{zone_index + 1}_Pa"
                    ] = youngs_modulus_chains[
                        chain_index,
                        :,
                        zone_index
                    ]

                pandas_frame = pd.DataFrame(
                    columns
                )

                pandas_frame.to_excel(
                    writer,
                    sheet_name=(
                        f"chain_{chain_index + 1:02d}"
                    ),
                    index=False
                )

            diagnostic_frame = pd.DataFrame(
                {
                    "chain": np.arange(
                        1,
                        chains.shape[0] + 1
                    ),
                    "acceptance_rate": (
                        acceptance_rates
                    ),
                    "warmup_acceptance_rate": (
                        warmup_acceptance_rates
                    )
                }
            )

            diagnostic_frame.to_excel(
                writer,
                sheet_name="diagnostics",
                index=False
            )

            pd.DataFrame(
                convergence_diagnostics
            ).to_excel(
                writer,
                sheet_name="convergence_diagnostics",
                index=False
            )

            pd.DataFrame(
                statistics
            ).to_excel(
                writer,
                sheet_name="posterior_statistics",
                index=False
            )