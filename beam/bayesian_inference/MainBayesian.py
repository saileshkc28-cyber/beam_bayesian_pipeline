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
            props = list({e.Properties.Id: e.Properties for e in part.Elements}.values())
            variable = Kratos.KratosGlobals.GetVariable(entry["control_variable"])
            # reference = the value StructuralMaterials.json assigned to this zone,
            # read once before any sample overwrites it. JSON may override it.
            ref = float(entry.get("reference_value") or props[0][variable])
            print(f"  {entry['name']}: {entry['target_sub_model_part']} "
                  f"({len(part.Elements)} elements), reference = {ref:.6e}")
            self.zones.append((variable, ref, props,
                               np.array([elem_at[e.Id] for e in part.Elements])))
        self.refs = np.array([z[1] for z in self.zones])
        self.prop_ids = [[p.Id for p in z[2]] for z in self.zones]

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

    os.makedirs("output/vtk_output", exist_ok=True)
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
        write_vtk(f"output/vtk_output/smc_level_{lev:02d}.vtk", model, disp, E)
        if lev == len(levels) - 1:
            write_vtk("output/vtk_output/posterior.vtk", model, disp, E)
    chain = levels[-1]
    E_ref = model.refs[0]
    E_chain = chain * E_ref              # same samples, physical units

    np.savez("output/smc_levels.npz", q=q, logcE=logcE, E_ref=E_ref,
             E_posterior=E_chain, prop_ids=np.array(model.prop_ids[0]),
             **{f"level_{i:02d}": lv for i, lv in enumerate(levels)})

    a = chain[:, 0]
    mean, std = a.mean(), a.std(ddof=1)
    Em, Es = E_chain.mean(), E_chain.std(ddof=1)
    json.dump({"alpha_mean": float(mean), "alpha_std": float(std),
               "E_mean_Pa": float(Em), "E_std_Pa": float(Es),
               "E_ref_Pa": E_ref, "logcE": logcE, "q": q.tolist(),
               "n_particles": int(len(a))},
              open("output/summary.json", "w"), indent=2)

    try:
        import pandas as pd
        with pd.ExcelWriter("output/posterior.xlsx", engine="openpyxl") as writer:
            for lev, samples in enumerate(levels):
                cols = {}
                for z in range(samples.shape[1]):
                    cols[f"alpha_{z + 1}"] = samples[:, z]
                    cols[f"E_{z + 1}_Pa"] = samples[:, z] * model.refs[z]
                pd.DataFrame(cols).to_excel(
                    writer, sheet_name=f"level_{lev:02d}_q{q[lev]:.3f}"[:31], index=False)
            pd.DataFrame([
                {"zone": z + 1, "alpha_mean": chain[:, z].mean(),
                 "alpha_std": chain[:, z].std(ddof=1),
                 "alpha_p2.5": np.percentile(chain[:, z], 2.5),
                 "alpha_p97.5": np.percentile(chain[:, z], 97.5),
                 "E_mean_Pa": chain[:, z].mean() * model.refs[z],
                 "E_std_Pa": chain[:, z].std(ddof=1) * model.refs[z]}
                for z in range(chain.shape[1])]).to_excel(
                    writer, sheet_name="posterior_stats", index=False)
    except ImportError:
        print("(pandas/openpyxl not installed - skipping posterior.xlsx)")

    print(f"\nalpha = {mean:.4f} +/- {std:.4f}")
    print(f"E     = {Em:.4e} +/- {Es:.3e} Pa   ({Em/1e9:.2f} +/- {Es/1e9:.2f} GPa)")
    print(f"logcE = {logcE:.3f}   levels = {len(q)}")