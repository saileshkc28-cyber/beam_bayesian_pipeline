"""Primal + adjoint Kratos model for the Phase 2 adjoint/Laplace inference.

Primal : the existing KratosForwardModel (unchanged), model part "Structure".
Adjoint: SystemIdentificationApplication's SystemIdentificationStaticAnalysis, the
         same adjoint machinery as DamageDetectionResponse / the CVaR setup.

How the two are connected (same idea as the CVaR OptimizationParameters.json, where
connectivity_preserving_model_part_controller couples Structure and AdjointStructure):

  1. the adjoint analysis is constructed first; its solver registers every nodal
     variable the adjoint needs (ADJOINT_DISPLACEMENT, ...);
  2. "Structure" is created with that same variable list, then the existing
     KratosForwardModel reads the mesh into it;
  3. "AdjointStructure" is generated from "Structure" with ConnectivityPreserveModeler:
     same node objects, same element ids, same Properties. The primal displacement
     field is therefore directly visible to the adjoint elements;
  4. the adjoint solver replaces the elements by their adjoint versions.

Sensitivities (sign convention of DamageDetectionResponse):
  dJ/dE_e = -(YOUNG_MODULUS_SENSITIVITY stored on element e)
  dJ/dalpha_i = E_ref,i * sum_{e in zone i} dJ/dE_e

Responses used with the adjoint:
  MeasurementResidualResponseFunction(p=1): J = sum_k 0.5 w_k (u_k - d_k)^2  (CVaR's J)
  each DisplacementSensor itself (it is an AdjointResponseFunction): du_k/dalpha
"""
import json

import numpy as np
import KratosMultiphysics as Kratos
import KratosMultiphysics.StructuralMechanicsApplication as KratosSMA
import KratosMultiphysics.OptimizationApplication as KratosOA
import KratosMultiphysics.SystemIdentificationApplication as KratosSI
from KratosMultiphysics.SystemIdentificationApplication.sensor_sensitivity_solvers.system_identification_static_analysis import SystemIdentificationStaticAnalysis
from KratosMultiphysics.SystemIdentificationApplication.utilities.sensor_utils import CreateSensors

from kratos_forward_model import KratosForwardModel


class AdjointSensitivityModel:

    def __init__(self, model, forward_settings, parameter_entries, adjoint_parameters_file,
                 element_name, sensor_model_part_name="AdjointLaplaceSensors"):
        forward_settings.ValidateAndAssignDefaults(KratosForwardModel.GetDefaultParameters())
        with open(forward_settings["primal_parameters_file"].GetString(), "r") as f:
            primal = Kratos.Parameters(f.read())
        with open(adjoint_parameters_file, "r") as f:
            adjoint = Kratos.Parameters(f.read())

        self.model = model
        self.primal_name = primal["solver_settings"]["model_part_name"].GetString()
        self.adjoint_name = adjoint["solver_settings"]["model_part_name"].GetString()
        for name in (self.primal_name, self.adjoint_name, sensor_model_part_name):
            if model.HasModelPart(name):
                raise RuntimeError(f"model part '{name}' already exists: pass a fresh Kratos.Model()")

        # 1) adjoint analysis first -> its solver adds all primal + adjoint variables
        self.adjoint_analysis = SystemIdentificationStaticAnalysis(model, adjoint)
        adjoint_mp = model[self.adjoint_name]

        # 2) primal root model part with the same nodal variable list
        primal_mp = model.CreateModelPart(self.primal_name)
        KratosOA.OptimizationUtils.SetSolutionStepVariablesList(primal_mp, adjoint_mp)
        primal_mp.ProcessInfo[Kratos.DOMAIN_SIZE] = \
            primal["solver_settings"]["domain_size"].GetInt()

        # 3) the existing, unchanged forward model reads the mesh into "Structure"
        self.forward = KratosForwardModel(model, forward_settings, parameter_entries)
        self.primal_mp = model[self.primal_name]

        # 4) adjoint model part shares the primal nodes (as in the CVaR setup)
        KratosOA.OptimizationUtils.SetSolutionStepVariablesList(adjoint_mp, self.primal_mp)
        Kratos.ConnectivityPreserveModeler().GenerateModelPart(self.primal_mp, adjoint_mp, element_name)
        sens_name = adjoint["solver_settings"]["sensitivity_settings"]["sensitivity_model_part_name"].GetString()
        if not adjoint_mp.HasSubModelPart(sens_name):
            smp = adjoint_mp.CreateSubModelPart(sens_name)
            smp.AddNodes([n.Id for n in adjoint_mp.Nodes])
            smp.AddElements([e.Id for e in adjoint_mp.Elements])

        # 5) adjoint initialisation: replaces elements by adjoint elements, adds dofs,
        #    applies the adjoint boundary conditions, builds the sensitivity builder
        self.adjoint_analysis.Initialize()
        self.adjoint_mp = model[self.adjoint_name]
        first = next(iter(self.adjoint_mp.Elements))
        Kratos.Logger.PrintInfo("AdjointSensitivityModel",
                                f"adjoint elements: {self.adjoint_mp.NumberOfElements()} x {first.Info()}")

        # zones on the adjoint side (same sub model part names under AdjointStructure)
        self.zone_parts = []
        for entry in parameter_entries:
            parts = entry["target_sub_model_part"].GetString().split(".")
            if parts[0] != self.primal_name:
                raise RuntimeError(f"zone '{'.'.join(parts)}' is not under '{self.primal_name}'")
            self.zone_parts.append(model[".".join([self.adjoint_name] + parts[1:])])
        self.refs = np.asarray(self.forward.refs, float)
        self.n_zones = len(self.zone_parts)

        zone_ids = [e.Id for mp in self.zone_parts for e in mp.Elements]
        self.zones_cover_all = (len(zone_ids) == len(set(zone_ids))
                                == self.primal_mp.NumberOfElements())

        # SI sensors located in the adjoint domain (same as sensor_model_part_controller)
        with open(forward_settings["sensor_data_file"].GetString(), "r") as f:
            sensor_list = json.load(f)["list_of_sensors"]
        self.sensor_mp = model.CreateModelPart(sensor_model_part_name)
        self.sensors = CreateSensors(self.sensor_mp, self.adjoint_mp,
                                     [Kratos.Parameters(json.dumps(s)) for s in sensor_list])
        self.sensor_names = [s.GetName() for s in self.sensors]
        self.sensor_weights = np.array([s.GetWeight() for s in self.sensors])
        for sensor in self.sensors:
            sensor.GetNode().SetValue(KratosSI.SENSOR_MEASURED_VALUE, 0.0)

        # CVaR's objective: MeasurementResidualResponseFunction with p = 1
        self.misfit = KratosSI.Responses.MeasurementResidualResponseFunction(1.0)
        for sensor in self.sensors:
            self.misfit.AddSensor(sensor)
        self.misfit.Initialize()

        self.sens_var = Kratos.KratosGlobals.GetVariable(
            Kratos.SensitivityUtilities.GetSensitivityVariableName(Kratos.YOUNG_MODULUS))
        self.sign = -1.0          # DamageDetectionResponse: dJ/dE = -(stored sensitivity)
        # adjoint unknowns, reset before every adjoint solve (see _RunAdjoint)
        self.adjoint_vars = [v for v in (KratosSMA.ADJOINT_DISPLACEMENT, KratosSMA.ADJOINT_ROTATION)
                             if self.adjoint_mp.HasNodalSolutionStepVariable(v)]
        self.n_adjoint = 0
        self.alpha = None

    # ----------------------------------------------------------------- counters
    @property
    def n_primal(self):
        return self.forward.n_solves

    # ------------------------------------------------------------------- primal
    def Evaluate(self, alpha):
        """Primal solve with the existing forward model; returns the sensor vector
        read with sensors.py (identical to Phase 1)."""
        alpha = np.atleast_1d(np.asarray(alpha, float))
        u = np.asarray(self.forward.Evaluate(alpha), float)
        # Properties are shared by the connectivity-preserving copy; set them on the
        # adjoint side as well so this never depends on that implementation detail.
        for a, ref, mp in zip(alpha, self.refs, self.zone_parts):
            for p in {e.Properties.Id: e.Properties for e in mp.Elements}.values():
                p.SetValue(Kratos.YOUNG_MODULUS, float(a) * ref)
        self.alpha = alpha
        return u

    def KratosSensorValues(self):
        """The same sensor values read by the Kratos DisplacementSensor objects."""
        return np.array([s.CalculateValue(self.primal_mp) for s in self.sensors])

    # ------------------------------------------------------------------ adjoint
    def _RunAdjoint(self, response, label):
        if self.alpha is None:
            raise RuntimeError("call Evaluate(alpha) before any adjoint solve")
        info = self.adjoint_analysis._GetSolver().GetComputingModelPart().ProcessInfo
        info[KratosSI.TEST_ANALYSIS_NAME] = "phase2_adjoint_laplace"
        try:
            info[KratosSI.SENSOR_NAME] = label
        except Exception:
            pass
        # The adjoint scheme solves in residual form, K^T dx = -dJ/du - K^T lambda_old,
        # and adds dx to the stored lambda. When responses of very different size follow
        # each other (a sensor adjoint, lambda ~ 1, then the misfit adjoint, lambda ~ 1e-9)
        # lambda_new is recovered by cancelling lambda_old and loses most of its digits.
        # Starting every solve from lambda = 0 removes that cancellation.
        for var in self.adjoint_vars:
            Kratos.VariableUtils().SetHistoricalVariableToZero(var, self.adjoint_mp.Nodes)
        self.adjoint_analysis.CalculateGradient(response)
        self.n_adjoint += 1
        g = np.zeros(self.n_zones)
        for i, (mp, ref) in enumerate(zip(self.zone_parts, self.refs)):
            g[i] = self.sign * ref * sum(e.GetValue(self.sens_var) for e in mp.Elements)
        return g

    def MisfitValueAndGradient(self, d):
        """CVaR/DamageDetectionResponse path: J = sum 0.5 w (u - d)^2 from Kratos and
        dJ/dalpha from ONE adjoint solve, at the last primal state."""
        for sensor, dk in zip(self.sensors, np.asarray(d, float)):
            sensor.GetNode().SetValue(KratosSI.SENSOR_MEASURED_VALUE, float(dk))
        J = self.misfit.CalculateValue(self.primal_mp)
        return float(J), self._RunAdjoint(self.misfit, "measurement_residual")

    def SensorJacobian(self):
        """S[k, i] = du_k/dalpha_i: one adjoint solve per sensor, at the last primal state."""
        S = np.zeros((len(self.sensors), self.n_zones))
        for k, sensor in enumerate(self.sensors):
            S[k] = self._RunAdjoint(sensor, sensor.GetName())
        return S

    # ------------------------------------------------------------------- checks
    def HomogeneityCheck(self, u, S, tolerance):
        """Exact identity for fixed load and zone-wise E covering the whole model:
        sum_i alpha_i du_k/dalpha_i = -u_k. Verifies sign and scale of the adjoint
        sensitivities without any finite difference. Returns (ratios, status)."""
        if not self.zones_cover_all:
            return None, "not applicable (zones do not cover every element exactly once)"
        u = np.asarray(u, float)
        lhs = S @ self.alpha
        valid = np.abs(u) > 1e-12 * np.max(np.abs(u))
        ratio = np.full(len(u), np.nan)
        ratio[valid] = lhs[valid] / (-u[valid])
        r = ratio[valid]
        if np.all(np.abs(r - 1.0) < tolerance):
            return ratio, "ok"
        if np.all(np.abs(r + 1.0) < tolerance):
            return ratio, "sign_flipped"
        return ratio, "failed"

    def Finalize(self):
        self.adjoint_analysis.Finalize()
        self.forward.Finalize()
