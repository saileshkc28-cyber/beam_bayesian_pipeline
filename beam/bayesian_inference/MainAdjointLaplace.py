"""Phase 2 runner: adjoint optimisation + Laplace posterior.

    python MainAdjointLaplace.py                      (uses AdjointLaplaceParameters.json)
    python MainAdjointLaplace.py MyParameters.json

Set "mode" in AdjointLaplaceParameters.json to "u_mean" (stage 1) or
"gauss_hermite" (stage 2). Model, zones and alpha prior are taken from the file in
"base_parameters_file" (BayesianParameters.json), so they match the SMC runs.
"""
import sys

import KratosMultiphysics as Kratos
from adjoint_laplace_analysis import RunAdjointLaplace

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "AdjointLaplaceParameters.json"
    with open(path, "r") as file_input:
        parameters = Kratos.Parameters(file_input.read())
    RunAdjointLaplace(parameters)
