import csv
import os
import json
import numpy as np
import KratosMultiphysics as Kratos
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis

import sensors as sensor_tools
from stats_output import FieldStats, write_vtk


class KratosForwardModel:
    """Build once, evaluate many. Initialize() is 1000x the cost of a solve."""

    def __init__(self, parameters, parameter_entries, sensor_file):
        self.model = Kratos.Model()
        self.analysis = StructuralMechanicsAnalysis(self.model, parameters)
        self.analysis.Initialize()
        self.root = self.model[parameters["solver_settings"]["model_part_name"].GetString()]
        self.t0 = parameters["problem_data"]["start_time"].GetDouble()

        elem_at = {e.Id: i for i, e in enumerate(self.root.Elements)}
        self.zones = []
        for entry in parameter_entries:
            part = self.model[entry["target_sub_model_part"]]
            props = {e.Properties.Id: e.Properties for e in part.Elements}
            self.zones.append((Kratos.KratosGlobals.GetVariable(entry["control_variable"]),
                               float(entry["reference_value"]), list(props.values()),
                               np.array([elem_at[e.Id] for e in part.Elements])))

        self.located = sensor_tools.locate(self.root, sensor_tools.read_sensors(sensor_file))
        self.nodes = np.array([[n.X0, n.Y0, n.Z0] for n in self.root.Nodes])
        idx = {n.Id: i for i, n in enumerate(self.root.Nodes)}
        self.cells = np.array([[idx[n.Id] for n in e.GetGeometry()] for e in self.root.Elements])
        self.field = None

    def evaluate(self, theta):
        for a, (variable, ref, props, _) in zip(np.atleast_1d(theta), self.zones):
            for p in props:
                p.SetValue(variable, float(a) * ref)

        # Clear() forces a clean rebuild of DOFs/LHS/RHS; without it state leaks between samples
        solver = self.analysis._GetSolver()
        solver.Clear()
        info = solver.GetComputingModelPart().ProcessInfo
        info[Kratos.STEP] = 0
        info[Kratos.TIME] = self.t0
        self.analysis.time = self.t0
        self.analysis.RunSolutionLoop()

        self.field = sensor_tools.displacements(self.root)
        return sensor_tools.read(self.field, self.located)

    def youngs_modulus(self, theta):
        E = np.zeros(len(self.cells))
        for a, (_, ref, _, elems) in zip(np.atleast_1d(theta), self.zones):
            E[elems] = float(a) * ref
        return E


def smc(model, u_hat, sigma, era_prior, settings, cache):
    """ERA SMC_aCS. Gets log-likelihood ONLY: the prior lives in the ERADist object."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "era"))
    from era.SMC_aCS import SMC_aCS

    def log_likelihood(theta):
        key = float(np.atleast_1d(theta)[0])
        if key not in cache:
            u = model.evaluate(theta)
            cache[key] = (u, model.field.copy())
        r = u_hat - cache[key][0]
        return float(-0.5 * float(r @ r) / sigma**2)   # plain float: numpy 2 rejects arrays

    np.random.seed(settings.get("random_seed"))
    _, samplesX, q, logcE = SMC_aCS(settings["n_particles"], settings["p"], log_likelihood,
                                    era_prior, settings["burn"], settings["target_cov"])
    levels = [np.atleast_2d(np.asarray(x, float)).reshape(-1, 1) for x in samplesX]
    return levels, np.asarray(q), float(logcE)


if __name__ == "__main__":

    os.makedirs("output", exist_ok=True)
    config = json.load(open("BayesianParameters.json"))

    with open(config["project_parameters_file"], "r") as file_input:
        parameters = Kratos.Parameters(file_input.read())

    model = KratosForwardModel(parameters, config["parameters"],
                               config["likelihood"]["sensor_data_file"])

    u_hat = np.array([float(r["value"]) for r in
                      csv.DictReader(open(config["likelihood"]["measured_data_file"]))])
    sigma = config["likelihood"]["noise_model"]["sigma"]

    from era.Distributions.ERADist import ERADist
    marginals = [ERADist(e["prior"]["type"], "PAR", e["prior"]["parameters"])
                 for e in config["parameters"]]

    settings = config["sampler_settings"]
    cache = {}
    levels, q, logcE = smc(model, u_hat, sigma, marginals[0],
                           {**settings["smc_acs"], "random_seed": settings["random_seed"]},
                           cache)
    for lev, thetas in enumerate(levels):
        disp, E = FieldStats(), FieldStats()
        for th in thetas:
            if float(th[0]) in cache:
                disp.update(cache[float(th[0])][1])
                E.update(model.youngs_modulus(th))
        write_vtk(f"output/smc_level_{lev:02d}.vtk", model, disp, E)
        if lev == len(levels) - 1:
            write_vtk("output/posterior.vtk", model, disp, E)
    chain = levels[-1]

    np.savez("output/smc_levels.npz", q=q, logcE=logcE,
             **{f"level_{i:02d}": lv for i, lv in enumerate(levels)})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    a = chain[:, 0]
    mean, std = a.mean(), a.std(ddof=1)
    ax.hist(a, bins=40, density=True, color="#4878a8", edgecolor="white",
            label=f"posterior samples (n={len(a)})")
    ax.axvline(mean, color="#1a3a5a", lw=1.8,
               label=rf"mean $\alpha$ = {mean:.4f} $\pm$ {std:.4f}")
    for s in (mean - std, mean + std):
        ax.axvline(s, color="#1a3a5a", lw=1.0, ls=":")

    truth_file = "../damaged_system/measurement_metadata.json"
    if os.path.exists(truth_file):
        truth = json.load(open(truth_file)).get("alpha_true")
        if truth is not None:
            ax.axvline(truth, color="crimson", ls="--", lw=1.5,
                       label=rf"$\alpha_{{true}}$ = {truth}")

    ax.set_xlabel(r"$\alpha = E/E_{ref}$")
    ax.set_ylabel("posterior density")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig("output/posterior_alpha.png", dpi=150)

    print(f"\nalpha = {chain.mean():.4f} +/- {chain.std(ddof=1):.4f}   "
          f"logcE = {logcE:.3f}   levels = {len(q)}")