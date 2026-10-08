"""Step 1 of the adjoint/Laplace Phase 2: verify the Kratos adjoint wiring only.

    python check_adjoint_setup.py

Builds the primal + adjoint model exactly as MainAdjointLaplace.py does, then runs
the three startup checks at alpha = 1.0 and at alpha = 0.8 (no optimisation):

  1) Kratos DisplacementSensor value == sensors.py value (same interpolation)
  2) sum_i alpha_i du/dalpha_i = -u   (exact for a fixed load; tests the adjoint
     sensitivities, sign and scale, without any finite difference)
  3) the Kratos misfit gradient (one adjoint) == the gradient rebuilt from the
     per-sensor adjoints

Every line should end in ok / a difference around 1e-8 or smaller.
"""
import sys

import numpy as np
import KratosMultiphysics as Kratos

from adjoint_laplace_analysis import ReadSettings, ReadBaseParameters, RunStartupChecks
from adjoint_sensitivity_model import AdjointSensitivityModel

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "AdjointLaplaceParameters.json"
    with open(path, "r") as f:
        parameters = Kratos.Parameters(f.read())
    settings = ReadSettings(parameters)
    forward_settings, entries = ReadBaseParameters(parameters)

    model = Kratos.Model()
    asm = AdjointSensitivityModel(model, forward_settings, entries,
                                  settings["adjoint_parameters_file"], settings["element_name"])
    print(f"sensors: {asm.sensor_names}")
    print(f"zones  : {[e['name'].GetString() for e in entries]}  E_ref = {asm.refs}")
    print(f"zones cover every element once: {asm.zones_cover_all}")

    for a in (1.0, 0.8):
        alpha = np.full(asm.n_zones, a)
        u = asm.Evaluate(alpha)
        d = 1.01 * u            # any target works for the gradient cross-check
        print(f"\n--- alpha = {alpha} ---")
        print(f"u = {u}")
        result = RunStartupChecks(asm, alpha, d, settings, print)
        S = asm.SensorJacobian()
        print(f"du/dalpha = {S.ravel()}   (for one zone this should equal -u/alpha = {(-u / a)})")

    print(f"\n{asm.n_primal} forward solves, {asm.n_adjoint} adjoint solves")
    asm.Finalize()
    print("adjoint setup OK")
