"""KratosForwardModel — build once, evaluate many.

The model, modelers and analysis stage are constructed a single time. Each call to
evaluate(theta) only writes theta into the cached Properties objects, re-solves,
and interpolates the displacement at the sensor points.

Nothing is hardcoded: the parameter -> Properties mapping, the reference values and
the sensor definitions all come from the JSON configuration. Adding a second zone
is a second entry under "parameters"; this class loops over entries.
"""
import time
import json
import numpy as np

import KratosMultiphysics as Kratos
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import (
    StructuralMechanicsAnalysis,
)


class KratosForwardModel:

    def __init__(self, project_parameters_file, parameter_entries, sensor_data_file,
                 self_check=True):
        # ---------------------------------------------------------- ONCE
        with open(project_parameters_file) as f:
            self.project_parameters = Kratos.Parameters(f.read())

        self.model = Kratos.Model()
        self.analysis = StructuralMechanicsAnalysis(self.model, self.project_parameters)
        self.analysis.Initialize()          # runs modelers: mdpa import + shell entities
        self._start_time = self.project_parameters["problem_data"]["start_time"].GetDouble()

        root_name = self.project_parameters["solver_settings"]["model_part_name"].GetString()
        self.root = self.model[root_name]

        # cache the Properties objects each parameter controls (per zone)
        self.parameter_entries = parameter_entries
        self.controlled = []                # list of (variable, reference_value, [Properties], [elem indices])
        elem_index = {e.Id: i for i, e in enumerate(self.root.Elements)}
        self.n_elements = len(elem_index)
        for entry in parameter_entries:
            smp = self.model[entry["target_sub_model_part"]]
            props, elem_ids = {}, []
            for elem in smp.Elements:
                props[elem.Properties.Id] = elem.Properties
                elem_ids.append(elem_index[elem.Id])
            variable = Kratos.KratosGlobals.GetVariable(entry["control_variable"])
            self.controlled.append(
                (variable, float(entry["reference_value"]), list(props.values()), np.array(elem_ids))
            )

        # mesh snapshot for VTK writing (node coords, connectivity as 0-based indices)
        node_index = {}
        coords = []
        for i, node in enumerate(self.root.Nodes):
            node_index[node.Id] = i
            coords.append([node.X0, node.Y0, node.Z0])
        self.mesh_nodes = np.array(coords)
        self.mesh_cells = np.array(
            [[node_index[n.Id] for n in e.GetGeometry()] for e in self.root.Elements]
        )
        self._node_index = node_index

        # cache sensor definitions and their interpolation data (element + barycentric weights)
        with open(sensor_data_file) as f:
            self.sensors = json.load(f)["list_of_sensors"]
        self._locate_sensors()

        self.n_solves = 0
        self.total_solve_time = 0.0
        self.last_displacement_field = None     # (n_nodes, 3), filled by evaluate()

        if self_check:
            self._self_check()

    # ------------------------------------------------------------ PER SAMPLE
    def evaluate(self, theta):
        """theta (physical space, one alpha per parameter entry) -> sensor value vector."""
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        if theta.size != len(self.controlled):
            raise ValueError(f"theta has {theta.size} entries, expected {len(self.controlled)}")

        t0 = time.perf_counter()
        # write theta into the Properties
        for a, (variable, ref, props, _) in zip(theta, self.controlled):
            for p in props:
                p.SetValue(variable, float(a) * ref)

        # re-solve: clear the strategy (forces a clean rebuild of DOF set, LHS and
        # RHS with the new properties — guards against every stale-state variant),
        # rewind the stage clock, and run the single static step again
        solver = self.analysis._GetSolver()
        solver.Clear()
        pi = solver.GetComputingModelPart().ProcessInfo
        pi[Kratos.STEP] = 0
        pi[Kratos.TIME] = self._start_time
        self.analysis.time = self._start_time
        self.analysis.RunSolutionLoop()

        # read displacements
        field = np.array(
            [list(n.GetSolutionStepValue(Kratos.DISPLACEMENT)) for n in self.root.Nodes]
        )
        self.last_displacement_field = field

        values = np.array(
            [float((w @ field[idx]) @ d) for idx, w, d in self._sensor_interp]
        )

        self.n_solves += 1
        self.total_solve_time += time.perf_counter() - t0
        return values

    def element_youngs_modulus(self, theta):
        """Per-element E vector implied by theta (no solve needed)."""
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        E = np.full(self.n_elements, np.nan)
        for a, (_, ref, _, elem_ids) in zip(theta, self.controlled):
            E[elem_ids] = float(a) * ref
        return E

    def mean_solve_time(self):
        return self.total_solve_time / max(self.n_solves, 1)

    # ------------------------------------------------------------ internals
    def _locate_sensors(self):
        """Find, once, the containing element and barycentric weights of each sensor."""
        self._sensor_interp = []    # (node indices[3], weights[3], direction[3])
        for s in self.sensors:
            px, py = s["location"][0], s["location"][1]
            direction = np.array(s["direction"], dtype=float)
            found = False
            for elem in self.root.Elements:
                nodes = list(elem.GetGeometry())
                x = np.array([n.X0 for n in nodes])
                y = np.array([n.Y0 for n in nodes])
                det = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
                l1 = ((y[1] - y[2]) * (px - x[2]) + (x[2] - x[1]) * (py - y[2])) / det
                l2 = ((y[2] - y[0]) * (px - x[2]) + (x[0] - x[2]) * (py - y[2])) / det
                l3 = 1.0 - l1 - l2
                if min(l1, l2, l3) >= -1e-10:
                    idx = np.array([self._node_index[n.Id] for n in nodes])
                    self._sensor_interp.append((idx, np.array([l1, l2, l3]), direction))
                    found = True
                    break
            if not found:
                raise RuntimeError(
                    f"sensor {s['name']} at ({px}, {py}) lies outside every element")

    def _self_check(self):
        """Guard against the stale-state bug: determinism, sensitivity, sensor sanity."""
        theta_a = np.ones(len(self.controlled))
        theta_b = 0.5 * np.ones(len(self.controlled))

        u1 = self.evaluate(theta_a)
        u2 = self.evaluate(theta_a)
        if not np.allclose(u1, u2, rtol=0, atol=1e-16 + 1e-12 * np.abs(u1).max()):
            raise RuntimeError(f"forward model is not deterministic: {u1} vs {u2}")

        u3 = self.evaluate(theta_b)
        if np.allclose(u1, u3, rtol=1e-6):
            raise RuntimeError(
                "changed theta did not change the output — properties are not "
                "propagating into the solve (stale-state bug)")

        if np.any(u1 == 0.0) or np.any(~np.isfinite(u1)):
            raise RuntimeError(f"sensor values contain zeros or NaN: {u1}")

        Kratos.Logger.PrintInfo(
            "KratosForwardModel",
            "self-check passed: deterministic, theta-sensitive, %d sensors nonzero; "
            "%.1f ms per evaluate" % (len(u1), 1e3 * self.mean_solve_time()))
