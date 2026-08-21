import importlib

import numpy as np
import KratosMultiphysics as Kratos

from kratos_forward_model import KratosForwardModel
from likelihood import Likelihood, Prior


class MCMCAnalysis:
    """Kratos-style analysis stage for MCMC parameter inference."""

    @staticmethod
    def GetDefaultParameters():
        return Kratos.Parameters("""{
            "problem_data": {
                "problem_name": "mcmc_inference",
                "echo_level": 1
            },
            "forward_model": {},
            "parameters": [],
            "likelihood": {},
            "sampler_settings": {
                "python_module": "metropolis_hastings_sampler",
                "Parameters": {}
            },
            "output_processes": []
        }""")

    def __init__(
        self,
        model,
        project_parameters
    ):
        project_parameters.ValidateAndAssignDefaults(
            self.GetDefaultParameters()
        )

        project_parameters[
            "problem_data"
        ].ValidateAndAssignDefaults(
            self.GetDefaultParameters()[
                "problem_data"
            ]
        )

        self.model = model
        self.settings = project_parameters

        self.echo_level = project_parameters[
            "problem_data"
        ]["echo_level"].GetInt()

        self.forward_model = None
        self.likelihood = None
        self.prior = None
        self.sampler = None
        self.output_processes = []

    def Initialize(self):
        parameter_entries = [
            self.settings["parameters"][index]
            for index in range(
                self.settings["parameters"].size()
            )
        ]

        if not parameter_entries:
            raise RuntimeError(
                "'parameters' is empty: nothing to infer"
            )

        self.forward_model = KratosForwardModel(
            self.model,
            self.settings["forward_model"],
            parameter_entries
        )

        self.likelihood = Likelihood(
            self.forward_model,
            self.settings["likelihood"]
        )

        self.prior = Prior(
            parameter_entries
        )

        sampler_settings = self.settings[
            "sampler_settings"
        ]

        sampler_module = importlib.import_module(
            "samplers."
            + sampler_settings[
                "python_module"
            ].GetString()
        )

        self.sampler = sampler_module.Factory(
            self.likelihood,
            self.prior,
            sampler_settings["Parameters"]
        )

        for index in range(
            self.settings["output_processes"].size()
        ):
            entry = self.settings[
                "output_processes"
            ][index]

            output_module = importlib.import_module(
                "processes."
                + entry[
                    "python_module"
                ].GetString()
            )

            self.output_processes.append(
                output_module.Factory(
                    self.forward_model,
                    self.likelihood,
                    entry["Parameters"]
                )
            )

    def RunSolutionLoop(self):
        Kratos.Logger.PrintInfo(
            "MCMCAnalysis",
            "Metropolis-Hastings sampling started"
        )

        posterior = self.sampler.Run()
        diagnostics = self.sampler.Diagnostics()

        acceptance_rates = np.asarray(
            diagnostics["acceptance_rates"],
            dtype=float
        )

        Kratos.Logger.PrintInfo(
            "MCMCAnalysis",
            (
                "sampling finished: "
                "%d chains, "
                "%d draws per chain, "
                "%d posterior samples, "
                "%d Kratos solves, "
                "mean acceptance = %.3f"
            )
            % (
                self.sampler.chains.shape[0],
                self.sampler.chains.shape[1],
                len(posterior),
                self.forward_model.n_solves,
                float(np.mean(acceptance_rates))
            )
        )

    def Finalize(self):
        try:
            for process in self.output_processes:
                process.ExecuteFinalize(
                    self.sampler
                )
        finally:
            self.forward_model.Finalize()

    def Run(self):
        self.Initialize()
        self.RunSolutionLoop()
        self.Finalize()