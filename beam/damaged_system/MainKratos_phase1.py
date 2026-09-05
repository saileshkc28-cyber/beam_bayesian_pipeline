import json
import os
import numpy as np
import KratosMultiphysics as Kratos
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis

from sensors import read_sensors, interpolate

# fixed noise floor, independent of the sensor set: 2% of the undamaged tip deflection
NOISE_SIGMA = 3.788e-08       # <<< USER INPUT REQUIRED: must equal the sigma used by Phase 2
SEED = 20260802

# ----------------------------------------------------------------------------- #
# Phase 1 distribution sampling                                                  #
# ----------------------------------------------------------------------------- #
N_SAMPLES = 1000      # start small, raise once the folder layout checks out
E_MEAN    = 206.9e9           # = E_ref -> alpha centered on 1.0
E_STD     = 20.69e9           # 0.10 * E_ref -> alpha sd 0.1, +-3sd spans 0.7-1.3
OUTPUT_DIRECTORY = "phase1_distribution_runs"

ENABLE_MEASUREMENT_NOISE = True   # NOISE_SIGMA <= 0 is an error while this is True
MAKE_PLOTS = True                 # histograms of xi, E and u into OUTPUT_DIRECTORY
PLOT_BINS = 30
WRITE_VTK_PER_SAMPLE = False      # False: VTK disabled; True: redirected per sample folder
RUN_SINGLE_DETERMINISTIC = False  # legacy behaviour; would overwrite ./measured_data.csv

PARAMETER_FILE = "PrimalParametersBeam.json"
MATERIAL_REFERENCE_FILE = "mat_ref.json"
SENSOR_DATA_PATH = "../sensor_placement/sensor_data.json"
PROPERTY_ID = 1
MEASURED_DATA_HEADER = "#,type,name,location_0,location_1,location_2,value\n"


class CustomStructuralMechanicsAnalysis(StructuralMechanicsAnalysis):
    """Copies YOUNG_MODULUS from Properties onto element data for the VTK block."""

    def FinalizeSolutionStep(self):

        for element in self._GetSolver().GetComputingModelPart().Elements:
            element.SetValue(Kratos.YOUNG_MODULUS, element.Properties[Kratos.YOUNG_MODULUS])

        super().FinalizeSolutionStep()


def read_reference_young_modulus(path=MATERIAL_REFERENCE_FILE, properties_id=PROPERTY_ID):
    """E_ref from mat_ref.json. Never touches StructuralMaterials.json."""
    with open(path, "r") as file_input:
        data = json.load(file_input)
    for block in data["properties"]:
        if block["properties_id"] == properties_id:
            return float(block["Material"]["Variables"]["YOUNG_MODULUS"])
    raise KeyError(f"no properties_id {properties_id} in {path}")


def run_forward(E_value, vtk_output_path=None):
    """One deterministic solve K(E) u = F. Returns the clean sensor displacement vector.

    E_value is applied to property PROPERTY_ID in memory after Initialize().
    StructuralMaterials.json is read but never written.
    E_value = None keeps whatever StructuralMaterials.json assigned (legacy behaviour).
    """
    model = Kratos.Model()

    with open(PARAMETER_FILE, "r") as file_input:
        parameters = Kratos.Parameters(file_input.read())

    if parameters["output_processes"].Has("sensor_output"):
        parameters["output_processes"].RemoveValue("sensor_output")

    # keep realizations from overwriting each other's VTK files
    if parameters["output_processes"].Has("vtk_output"):
        if vtk_output_path is None:
            parameters["output_processes"].RemoveValue("vtk_output")
        else:
            for vtk_settings in parameters["output_processes"]["vtk_output"]:
                vtk_settings["Parameters"]["output_path"].SetString(vtk_output_path)

    analysis = CustomStructuralMechanicsAnalysis(model, parameters)
    analysis.Initialize()          # materials are read here, so override afterwards

    structure = model["Structure"]

    if E_value is not None:
        for element in structure.Elements:
            if element.Properties.Id == PROPERTY_ID:
                element.Properties.SetValue(Kratos.YOUNG_MODULUS, float(E_value))

        applied = {e.Properties.Id: e.Properties[Kratos.YOUNG_MODULUS]
                   for e in structure.Elements}
        if PROPERTY_ID not in applied:
            raise RuntimeError(f"no element uses property {PROPERTY_ID}")
        if abs(applied[PROPERTY_ID] - float(E_value)) > 1e-6 * abs(float(E_value)):
            raise RuntimeError(f"property {PROPERTY_ID} holds {applied[PROPERTY_ID]:.6e} Pa, "
                               f"expected {float(E_value):.6e} Pa")

    analysis.RunSolutionLoop()
    analysis.Finalize()

    for pid, E in sorted({e.Properties.Id: e.Properties[Kratos.YOUNG_MODULUS]
                          for e in structure.Elements}.items()):
        print(f"property {pid}: E_true = {E:.6e} Pa")

    sensors = read_sensors(SENSOR_DATA_PATH)
    return interpolate(structure, sensors)


def write_measured_data(path, sensors, u_hat):
    """Exactly the header and row layout the existing Phase 2 likelihood reads."""
    with open(path, "w") as f:
        f.write(MEASURED_DATA_HEADER)
        for i, (s, v) in enumerate(zip(sensors, u_hat), 1):
            f.write(f"{i},{s['type']},{s['name']},{s['location'][0]},{s['location'][1]},"
                    f"{s['location'][2]},{v:.16e}\n")


def _normal_pdf(x, mean, sd):
    return np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * np.sqrt(2.0 * np.pi))


def make_plots(xi_values, E_values, u_matrix, sensors, out_dir):
    """Three histograms: xi, E(xi) and the sensor response u. Returns written paths."""
    try:
        import matplotlib
        matplotlib.use("Agg")          # no display needed
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available - skipping plots "
              "(all data is still in phase1_samples.csv)")
        return []

    written = []

    def histogram(ax, data, label, mean=None, sd=None):
        bins = min(PLOT_BINS, max(5, len(data) // 3))   # 20 samples in 30 bins is unreadable
        ax.hist(data, bins=bins, density=True, edgecolor="black", alpha=0.75)
        if mean is not None and sd is not None and sd > 0.0:
            grid = np.linspace(data.min(), data.max(), 400)
            ax.plot(grid, _normal_pdf(grid, mean, sd), "r-", lw=1.5, label="target normal")
            ax.legend(fontsize=8)
        ax.axvline(data.mean(), color="k", ls="--", lw=1.0)
        ax.set_xlabel(label)
        ax.set_ylabel("density")

    # 1) xi ~ N(0,1)
    fig, ax = plt.subplots(figsize=(6, 4))
    histogram(ax, xi_values, r"$\xi$", 0.0, 1.0)
    ax.set_title(f"xi: n = {len(xi_values)}, mean = {xi_values.mean():.4f}, "
                 f"sd = {xi_values.std(ddof=1):.4f}")
    fig.tight_layout()
    path = os.path.join(out_dir, "dist_xi.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    # 2) E = E_mean + E_std * xi
    fig, ax = plt.subplots(figsize=(6, 4))
    histogram(ax, E_values, "E [Pa]", float(E_MEAN), float(E_STD))
    ax.set_title(f"E: mean = {E_values.mean():.4e}, sd = {E_values.std(ddof=1):.4e} Pa")
    fig.tight_layout()
    path = os.path.join(out_dir, "dist_E.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    # 3) u, one panel per sensor (propagated, not assumed normal)
    n = len(sensors)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    for j, s in enumerate(sensors):
        column = u_matrix[:, j]
        histogram(axes[0][j], column, f"u [{s['name']}]")
        axes[0][j].set_title(f"{s['name']}: mean = {column.mean():.4e}, "
                             f"sd = {column.std(ddof=1):.4e}")
    fig.tight_layout()
    path = os.path.join(out_dir, "dist_u.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    return written


def run_single_deterministic():
    """Original Phase 1 run: writes ./measured_data.csv and ./noise_model.json."""
    sensors = read_sensors(SENSOR_DATA_PATH)
    u_true = run_forward(None, vtk_output_path="vtk_output")

    sigma = NOISE_SIGMA
    u_hat = u_true + np.random.default_rng(SEED).normal(0.0, sigma, u_true.shape)

    write_measured_data("measured_data.csv", sensors, u_hat)
    with open("noise_model.json", "w") as f:
        json.dump({"sigma": sigma, "seed": SEED,
                   "u_true": u_true.tolist(), "u_hat": u_hat.tolist()}, f, indent=2)

    print(f"sigma = {sigma:.6e}")
    for s, ut, uh in zip(sensors, u_true, u_hat):
        print(f"{s['name']}: u_true = {ut: .6e}   u_hat = {uh: .6e}")


def run_distribution():
    missing = [name for name, value in
               (("N_SAMPLES", N_SAMPLES), ("E_MEAN", E_MEAN), ("E_STD", E_STD))
               if value is None]
    if missing:
        raise ValueError("set these constants before running: " + ", ".join(missing))
    if int(N_SAMPLES) < 1:
        raise ValueError(f"N_SAMPLES must be >= 1, got {N_SAMPLES}")
    if float(E_STD) < 0.0:
        raise ValueError(f"E_STD must be >= 0, got {E_STD}")
    if ENABLE_MEASUREMENT_NOISE and NOISE_SIGMA <= 0:
        raise ValueError(f"measurement noise is enabled but NOISE_SIGMA = {NOISE_SIGMA}; "
                         "set it to the sigma Phase 2 uses")

    E_ref = read_reference_young_modulus()
    sensors = read_sensors(SENSOR_DATA_PATH)
    sigma = float(NOISE_SIGMA)

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    # one generator for the whole experiment -> the seed reproduces E_i and the noise
    rng = np.random.default_rng(SEED)
    xi_samples = rng.standard_normal(int(N_SAMPLES))

    E_samples = []
    for i, xi in enumerate(xi_samples, 1):
        E_i = float(E_MEAN) + float(E_STD) * float(xi)
        if E_i <= 0.0:
            raise ValueError(f"sample {i:04d} rejected: xi = {xi:+.6f} gives "
                             f"E = {E_i:.6e} Pa <= 0. Reduce E_STD or raise E_MEAN; "
                             "the value is not clipped.")
        E_samples.append(E_i)

    rows = []
    failures = []
    for i, (xi, E_i) in enumerate(zip(xi_samples, E_samples), 1):
        alpha_i = E_i / E_ref
        sample_dir = os.path.join(OUTPUT_DIRECTORY, f"sample_{i:04d}")
        os.makedirs(sample_dir, exist_ok=True)

        print(f"\n--- sample {i:04d} / {int(N_SAMPLES)}: xi = {float(xi):+.6f}, "
              f"E = {E_i:.6e} Pa, alpha = {alpha_i:.6f} ---")

        vtk_path = os.path.join(sample_dir, "vtk_output") if WRITE_VTK_PER_SAMPLE else None
        noise = rng.normal(0.0, sigma, len(sensors)) if ENABLE_MEASUREMENT_NOISE \
            else np.zeros(len(sensors))

        try:
            u_true = run_forward(E_i, vtk_output_path=vtk_path)
            status = "ok"
        except Exception as exc:                       # keep the loop going, report at the end
            status = f"failed: {type(exc).__name__}: {exc}"
            failures.append((i, status))
            print(status)
            rows.append([i, float(xi), E_i, alpha_i,
                         [float("nan")] * len(sensors), [float("nan")] * len(sensors), status])
            continue

        u_hat = u_true + noise
        for s, ut, uh in zip(sensors, u_true, u_hat):
            print(f"  {s['name']}: u_true = {ut: .6e}   u_hat = {uh: .6e}")

        write_measured_data(os.path.join(sample_dir, "measured_data.csv"), sensors, u_hat)

        with open(os.path.join(sample_dir, "noise_model.json"), "w") as f:
            json.dump({"sigma": sigma, "seed": SEED,
                       "u_true": u_true.tolist(), "u_hat": u_hat.tolist()}, f, indent=2)

        with open(os.path.join(sample_dir, "measurement_metadata.json"), "w") as f:
            json.dump({"sample_id": i,
                       "xi": float(xi),
                       "E_true": E_i,
                       "E_ref": E_ref,
                       "alpha_true": alpha_i,
                       "E_distribution_mean": float(E_MEAN),
                       "E_distribution_sd": float(E_STD),
                       "sigma": sigma,
                       "seed": SEED,
                       "u_true": u_true.tolist(),
                       "u_hat": u_hat.tolist()}, f, indent=2)

        rows.append([i, float(xi), E_i, alpha_i, u_true.tolist(), u_hat.tolist(), status])

    aggregate = os.path.join(OUTPUT_DIRECTORY, "phase1_samples.csv")
    with open(aggregate, "w") as f:
        header = ["sample_id", "xi", "E_true", "alpha_true"]
        header += [f"u_true_{s['name']}" for s in sensors]
        header += [f"u_hat_{s['name']}" for s in sensors]
        header += ["status"]
        f.write(",".join(header) + "\n")
        for sid, xi, E_i, alpha_i, u_true, u_hat, status in rows:
            values = [str(sid), f"{xi:.16e}", f"{E_i:.16e}", f"{alpha_i:.16e}"]
            values += [f"{v:.16e}" for v in u_true]
            values += [f"{v:.16e}" for v in u_hat]
            values += ['"' + status + '"']
            f.write(",".join(values) + "\n")

    E_array = np.array(E_samples)
    ok_rows = [r for r in rows if r[6] == "ok"]

    # per-sample intermediate values
    print("\n---------------- xi -> E -> u ----------------")
    head = f"{'id':>5} {'xi':>12} {'E [Pa]':>14} {'alpha':>10}"
    head += "".join(f" {'u_' + s['name']:>16}" for s in sensors)
    print(head)
    for sid, xi, E_i, alpha_i, u_true, u_hat, status in rows:
        line = f"{sid:5d} {xi:12.6f} {E_i:14.6e} {alpha_i:10.6f}"
        line += "".join(f" {v:16.6e}" for v in u_true)
        print(line if status == "ok" else line + "   " + status)

    print("\n================ Phase 1 sampling summary ================")
    print(f"samples          : {int(N_SAMPLES)}   seed = {SEED}")
    print(f"E requested      : mean = {float(E_MEAN):.6e} Pa   sd = {float(E_STD):.6e} Pa")
    print(f"E empirical      : mean = {E_array.mean():.6e} Pa   "
          f"sd = {E_array.std(ddof=1) if len(E_array) > 1 else 0.0:.6e} Pa")
    print(f"E_ref            : {E_ref:.6e} Pa")
    print(f"alpha empirical  : mean = {(E_array / E_ref).mean():.6f}")
    print(f"sigma            : {sigma:.6e}   noise enabled = {ENABLE_MEASUREMENT_NOISE}")
    print(f"aggregate        : {aggregate}")

    if ok_rows:
        u_matrix = np.array([r[4] for r in ok_rows])
        xi_ok = np.array([r[1] for r in ok_rows])
        E_ok = np.array([r[2] for r in ok_rows])
        for j, s in enumerate(sensors):
            column = u_matrix[:, j]
            print(f"u [{s['name']}]  : mean = {column.mean():.6e}   "
                  f"sd = {column.std(ddof=1) if len(column) > 1 else 0.0:.6e}")
        if MAKE_PLOTS:
            for path in make_plots(xi_ok, E_ok, u_matrix, sensors, OUTPUT_DIRECTORY):
                print(f"plot             : {path}")

    if failures:
        raise RuntimeError(f"{len(failures)} of {int(N_SAMPLES)} realizations failed; "
                           f"see the status column in {aggregate}")


if __name__ == "__main__":

    if RUN_SINGLE_DETERMINISTIC:
        run_single_deterministic()
    else:
        run_distribution()