import os

import numpy as np
import KratosMultiphysics as Kratos


def Factory(
    forward_model,
    likelihood,
    settings
):
    return MCMCVtkStatisticsOutputProcess(
        forward_model,
        likelihood,
        settings
    )


class FieldStatistics:
    """Calculate running mean and standard deviation using Welford's method."""

    def __init__(self):
        self.count = 0
        self.mean = None
        self.sum_squared_difference = None

    def Update(self, field):
        field = np.asarray(
            field,
            dtype=float
        )

        if self.mean is None:
            self.mean = np.zeros_like(field)
            self.sum_squared_difference = (
                np.zeros_like(field)
            )

        self.count += 1

        difference = field - self.mean
        self.mean += difference / self.count

        self.sum_squared_difference += (
            difference
            * (field - self.mean)
        )

    def Mean(self):
        if self.count == 0:
            raise RuntimeError(
                "cannot calculate field mean without samples"
            )

        return self.mean

    def Std(self):
        if self.count == 0:
            raise RuntimeError(
                "cannot calculate field standard deviation "
                "without samples"
            )

        if self.count == 1:
            return np.zeros_like(self.mean)

        return np.sqrt(
            self.sum_squared_difference
            / (self.count - 1)
        )


class MCMCVtkStatisticsOutputProcess:
    """Write posterior displacement and Young's-modulus statistics to VTK."""

    @staticmethod
    def GetDefaultParameters():
        return Kratos.Parameters("""{
            "python_module": "mcmc_vtk_statistics_output_process",
            "Parameters": {
                "output_path": "output_mcmc/vtk_output",
                "output_file_name": "posterior_mcmc.vtk"
            }
        }""")

    def __init__(
        self,
        forward_model,
        likelihood,
        settings
    ):
        settings.ValidateAndAssignDefaults(
            self.GetDefaultParameters()["Parameters"]
        )

        self.model = forward_model
        self.likelihood = likelihood

        self.output_path = settings[
            "output_path"
        ].GetString()

        self.output_file_name = settings[
            "output_file_name"
        ].GetString()

    def ExecuteFinalize(self, sampler):
        posterior = np.asarray(
            sampler.Posterior(),
            dtype=float
        )

        if posterior.ndim != 2:
            raise ValueError(
                "MCMC posterior must have shape "
                "(samples, parameters)"
            )

        if posterior.shape[0] == 0:
            raise RuntimeError(
                "MCMC posterior contains no samples"
            )

        displacement_statistics = (
            FieldStatistics()
        )

        youngs_modulus_statistics = (
            FieldStatistics()
        )

        for theta in posterior:
            cached_result = self.likelihood.Cached(
                theta
            )

            if cached_result is None:
                raise RuntimeError(
                    "a posterior sample is missing from the "
                    "likelihood cache; VTK export will not "
                    "start additional Kratos solves"
                )

            displacement_field = cached_result[1]

            displacement_statistics.Update(
                displacement_field
            )

            youngs_modulus_statistics.Update(
                self.model.YoungsModulus(theta)
            )

        os.makedirs(
            self.output_path,
            exist_ok=True
        )

        output_file = os.path.join(
            self.output_path,
            self.output_file_name
        )

        self._WriteVtk(
            output_file,
            displacement_statistics,
            youngs_modulus_statistics,
            posterior.shape[0]
        )

        Kratos.Logger.PrintInfo(
            "MCMCVtkStatisticsOutput",
            (
                "wrote posterior field statistics from "
                "%d samples to %s"
            )
            % (
                posterior.shape[0],
                output_file
            )
        )

    def _WriteVtk(
        self,
        output_file,
        displacement_statistics,
        youngs_modulus_statistics,
        number_of_samples
    ):
        nodes = np.asarray(
            self.model.nodes,
            dtype=float
        )

        cells = np.asarray(
            self.model.cells,
            dtype=int
        )

        if cells.ndim != 2 or cells.shape[1] != 3:
            raise ValueError(
                "the current VTK writer expects "
                "three-node triangular cells"
            )

        with open(
            output_file,
            "w"
        ) as vtk_file:
            vtk_file.write(
                "# vtk DataFile Version 4.2\n"
            )

            vtk_file.write(
                "MCMC posterior statistics from "
                f"{number_of_samples} samples\n"
            )

            vtk_file.write(
                "ASCII\n"
                "DATASET UNSTRUCTURED_GRID\n"
            )

            vtk_file.write(
                f"POINTS {len(nodes)} double\n"
            )

            for point in nodes:
                vtk_file.write(
                    "%.10e %.10e %.10e\n"
                    % (
                        point[0],
                        point[1],
                        point[2]
                    )
                )

            vtk_file.write(
                "\nCELLS %d %d\n"
                % (
                    len(cells),
                    4 * len(cells)
                )
            )

            for cell in cells:
                vtk_file.write(
                    "3 %d %d %d\n"
                    % (
                        cell[0],
                        cell[1],
                        cell[2]
                    )
                )

            vtk_file.write(
                f"\nCELL_TYPES {len(cells)}\n"
            )

            vtk_file.write(
                "5\n" * len(cells)
            )

            vtk_file.write(
                f"\nPOINT_DATA {len(nodes)}\n"
            )

            self._WriteVectorField(
                vtk_file,
                "DISPLACEMENT_MEAN",
                displacement_statistics.Mean()
            )

            self._WriteVectorField(
                vtk_file,
                "DISPLACEMENT_STD",
                displacement_statistics.Std()
            )

            vtk_file.write(
                f"\nCELL_DATA {len(cells)}\n"
            )

            self._WriteScalarField(
                vtk_file,
                "YOUNG_MODULUS_MEAN",
                youngs_modulus_statistics.Mean()
            )

            self._WriteScalarField(
                vtk_file,
                "YOUNG_MODULUS_STD",
                youngs_modulus_statistics.Std()
            )

    @staticmethod
    def _WriteVectorField(
        vtk_file,
        name,
        values
    ):
        values = np.asarray(
            values,
            dtype=float
        )

        vtk_file.write(
            f"VECTORS {name} double\n"
        )

        for value in values:
            vtk_file.write(
                "%.10e %.10e %.10e\n"
                % (
                    value[0],
                    value[1],
                    value[2]
                )
            )

    @staticmethod
    def _WriteScalarField(
        vtk_file,
        name,
        values
    ):
        values = np.asarray(
            values,
            dtype=float
        ).reshape(-1)

        vtk_file.write(
            f"SCALARS {name} double 1\n"
        )

        vtk_file.write(
            "LOOKUP_TABLE default\n"
        )

        for value in values:
            vtk_file.write(
                "%.10e\n" % value
            )