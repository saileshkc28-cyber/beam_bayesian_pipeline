import json
import numpy as np
import KratosMultiphysics as Kratos
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis

from sensors import read_sensors, interpolate

NOISE_FRACTION = 0.02
SEED = 20260811
E_REF = 69.0e9

ZONES = ["Region_1_alpha_100", "Region_2_alpha_50",
         "Region_3_alpha_080", "Region_4_alpha_100"]


class CustomStructuralMechanicsAnalysis(StructuralMechanicsAnalysis):
    """Copies YOUNG_MODULUS from Properties onto element data for the VTK block."""

    def FinalizeSolutionStep(self):

        for element in self._GetSolver().GetComputingModelPart().Elements:
            element.SetValue(Kratos.YOUNG_MODULUS, element.Properties[Kratos.YOUNG_MODULUS])

        super().FinalizeSolutionStep()


if __name__ == "__main__":

    model = Kratos.Model()

    with open("PrimalParametersPlate.json", "r") as file_input:
        parameters = Kratos.Parameters(file_input.read())

    analysis = CustomStructuralMechanicsAnalysis(model, parameters)
    analysis.Run()

    # the planted truth is whatever StructuralMaterials.json assigned, read back per zone
    E_true = {}
    for zone in ZONES:
        part = model["Structure." + zone]
        E_true[zone] = next(iter(part.Elements)).Properties[Kratos.YOUNG_MODULUS]

    sensors = read_sensors("../sensor_placement/sensor_data.json")
    u_true = interpolate(model["Structure"], sensors)

    sigma = NOISE_FRACTION * np.max(np.abs(u_true))
    u_hat = u_true + np.random.default_rng(SEED).normal(0.0, sigma, u_true.shape)

    with open("measured_data.csv", "w") as f:
        f.write("#,type,name,location_0,location_1,location_2,value\n")
        for i, (s, v) in enumerate(zip(sensors, u_hat), 1):
            f.write(f"{i},{s['type']},{s['name']},{s['location'][0]},{s['location'][1]},"
                    f"{s['location'][2]},{v:.16e}\n")

    with open("measurement_metadata.json", "w") as f:
        json.dump({"E_ref": E_REF,
                   "alpha_true": {z: E_true[z] / E_REF for z in ZONES},
                   "sigma": sigma,
                   "seed": SEED,
                   "u_true": u_true.tolist(),
                   "u_hat": u_hat.tolist()}, f, indent=2)

    for zone in ZONES:
        print(f"{zone}: E_true = {E_true[zone]:.6e} Pa   alpha = {E_true[zone] / E_REF:.4f}")
    print(f"sigma = {sigma:.6e}")
    for s, ut, uh in zip(sensors, u_true, u_hat):
        print(f"{s['name']}: u_true = {ut: .6e}   u_hat = {uh: .6e}")
