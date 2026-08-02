"""Step 0 — generate synthetic noisy measurements for the beam Bayesian pipeline.

Runs the Phase-1 forward solve at alpha_true = E_true / E_ref, interpolates the
displacement at the 4 sensor points (barycentric interpolation inside the
containing triangle, dotted with the sensor direction), adds Gaussian noise,
and writes measured_data.csv in the same format the plate project uses.

The sensor interpolation here is deliberately the same routine the Phase-2
KratosForwardModel will use, so measurement generation and inversion read
sensors identically. Does not require SystemIdentificationApplication.
"""
import json
import numpy as np
import KratosMultiphysics as Kratos
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import (
    StructuralMechanicsAnalysis,
)

# ------------------------------------------------------------------ settings
E_REF = 206.9e9
ALPHA_TRUE = 0.7
NOISE_FRACTION = 0.02      # sigma = 2 % of the largest |sensor value|
SEED = 20260802

SENSOR_FILE = "../sensor_placement/sensor_data.json"
PARAMS_FILE = "PrimalParametersBeam.json"
OUT_CSV = "measured_data.csv"
OUT_META = "measurement_metadata.json"


# ------------------------------------------------- sensor interpolation
def interpolate_sensors(model_part, sensors):
    """Barycentric interpolation of DISPLACEMENT at each sensor point."""
    values = []
    for s in sensors:
        px, py = s["location"][0], s["location"][1]
        direction = np.array(s["direction"])
        hit = None
        for elem in model_part.Elements:
            nodes = elem.GetGeometry()
            x = np.array([n.X for n in nodes])
            y = np.array([n.Y for n in nodes])
            det = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
            l1 = ((y[1] - y[2]) * (px - x[2]) + (x[2] - x[1]) * (py - y[2])) / det
            l2 = ((y[2] - y[0]) * (px - x[2]) + (x[0] - x[2]) * (py - y[2])) / det
            l3 = 1.0 - l1 - l2
            eps = 1e-10
            if l1 >= -eps and l2 >= -eps and l3 >= -eps:
                disp = np.zeros(3)
                for lam, node in zip((l1, l2, l3), nodes):
                    d = node.GetSolutionStepValue(Kratos.DISPLACEMENT)
                    disp += lam * np.array([d[0], d[1], d[2]])
                hit = float(disp @ direction)
                break
        if hit is None:
            raise RuntimeError(f"sensor {s['name']} at ({px},{py}) is outside every element")
        values.append(hit)
    return np.array(values)


# ------------------------------------------------------------------ run
with open(PARAMS_FILE) as f:
    parameters = Kratos.Parameters(f.read())

# strip output processes not needed for measurement generation
parameters["output_processes"] = Kratos.Parameters("""{}""")

model = Kratos.Model()
analysis = StructuralMechanicsAnalysis(model, parameters)
analysis.Run()

structure = model["Structure"]

# sanity: confirm the truth E actually reached the Properties
E_in_props = structure.GetSubModelPart("Beam_Auto1").Elements.__iter__().__next__().Properties[Kratos.YOUNG_MODULUS]
assert abs(E_in_props - ALPHA_TRUE * E_REF) / E_REF < 1e-12, f"E in Properties = {E_in_props}"

sensors = json.load(open(SENSOR_FILE))["list_of_sensors"]
u_true = interpolate_sensors(structure, sensors)

# noise
sigma = NOISE_FRACTION * np.max(np.abs(u_true))
rng = np.random.default_rng(SEED)
noise = rng.normal(0.0, sigma, size=u_true.shape)
u_hat = u_true + noise

# write CSV, same format as the plate project's measured_data.csv
with open(OUT_CSV, "w") as f:
    f.write("#,type,name,location_0,location_1,location_2,value\n")
    for i, (s, v) in enumerate(zip(sensors, u_hat), 1):
        L = s["location"]
        f.write(f"{i},{s['type']},{s['name']},{L[0]},{L[1]},{L[2]},{v:.16e}\n")

meta = {
    "alpha_true": ALPHA_TRUE,
    "E_ref": E_REF,
    "E_true": ALPHA_TRUE * E_REF,
    "noise_fraction_of_max": NOISE_FRACTION,
    "sigma": sigma,
    "rng_seed": SEED,
    "sensor_names": [s["name"] for s in sensors],
    "u_true_noiseless": u_true.tolist(),
    "noise_realisation": noise.tolist(),
    "u_hat_measured": u_hat.tolist(),
}
json.dump(meta, open(OUT_META, "w"), indent=2)

print("\nsigma =", sigma)
for s, ut, uh in zip(sensors, u_true, u_hat):
    print(f"{s['name']}: u_true = {ut: .6e}   u_hat = {uh: .6e}")
