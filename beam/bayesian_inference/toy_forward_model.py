"""ToyForwardModel — validation stub with the same interface as KratosForwardModel.

Implements u(alpha) = u_ref / alpha using the reference sensor values recovered from
the Step-0 metadata (u_ref = alpha_true * u_true_noiseless, exact for the linear
single-zone case). It exists ONLY to validate the sampler layer cheaply; it is not
part of the implementation path of the real pipeline and dies at multi-zone.
"""
import json
import numpy as np


class ToyForwardModel:

    def __init__(self, metadata_file):
        meta = json.load(open(metadata_file))
        self.u_ref = np.array(meta["u_true_noiseless"]) * meta["alpha_true"]
        self.n_solves = 0
        self.total_solve_time = 0.0
        self.last_displacement_field = None      # no field: VTK is skipped in toy mode
        self.mesh_nodes = None
        self.mesh_cells = None
        self.n_elements = 0

    def evaluate(self, theta):
        theta = np.atleast_1d(np.asarray(theta, dtype=float))
        if theta.size != 1:
            raise ValueError("the toy model is single-zone by construction")
        self.n_solves += 1
        return self.u_ref / theta[0]

    def element_youngs_modulus(self, theta):
        return np.array([])

    def mean_solve_time(self):
        return 0.0
