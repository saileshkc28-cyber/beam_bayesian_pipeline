"""Minimal legacy ASCII VTK writer for posterior field statistics.

Why not VtkOutputProcess: it can only output *registered* (compiled) Kratos
variables, and DISPLACEMENT_MEAN / DISPLACEMENT_STD / YOUNG_MODULUS_MEAN /
YOUNG_MODULUS_STD do not exist as registered variables. Writing them under
borrowed registered names (VELOCITY, ...) would produce misleading thesis
figures. This writer emits the same triangle mesh with correctly named arrays;
the files load in ParaView next to the deterministic Phase-2 VTK output.

The statistics passed in must come from the sample accumulator, never from a
solve at the posterior-mean theta: u(E[alpha]) != E[u(alpha)].
"""
import numpy as np


def write_vtk(path, nodes, cells, point_data=None, cell_data=None, title="posterior statistics"):
    """nodes: (n,3); cells: (m,3) 0-based triangle connectivity.
    point_data / cell_data: dict name -> (n,3) vector or (n,) scalar array."""
    nodes = np.asarray(nodes, float)
    cells = np.asarray(cells, int)
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 4.2\n")
        f.write(title + "\n")
        f.write("ASCII\nDATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {len(nodes)} double\n")
        for p in nodes:
            f.write(f"{p[0]:.10e} {p[1]:.10e} {p[2]:.10e}\n")
        f.write(f"\nCELLS {len(cells)} {4 * len(cells)}\n")
        for c in cells:
            f.write(f"3 {c[0]} {c[1]} {c[2]}\n")
        f.write(f"\nCELL_TYPES {len(cells)}\n")
        f.write("5\n" * len(cells))

        def _block(header_written, data, count, kind):
            first = not header_written
            for name, arr in data.items():
                arr = np.asarray(arr, float)
                if first:
                    f.write(f"\n{kind} {count}\n")
                    first = False
                if arr.ndim == 2:
                    f.write(f"VECTORS {name} double\n")
                    for v in arr:
                        f.write(f"{v[0]:.10e} {v[1]:.10e} {v[2]:.10e}\n")
                else:
                    f.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n")
                    for v in arr:
                        f.write(f"{v:.10e}\n")

        if point_data:
            _block(False, point_data, len(nodes), "POINT_DATA")
        if cell_data:
            _block(False, cell_data, len(cells), "CELL_DATA")


def write_stats_vtk(path, forward_model, disp_stats, E_stats, title):
    """Convenience: write DISPLACEMENT_MEAN/STD (nodal) + YOUNG_MODULUS_MEAN/STD
    (element) from two FieldStats accumulators. Skipped silently in toy mode."""
    if forward_model.mesh_nodes is None or disp_stats.mean() is None:
        return False
    write_vtk(
        path,
        forward_model.mesh_nodes,
        forward_model.mesh_cells,
        point_data={"DISPLACEMENT_MEAN": disp_stats.mean(),
                    "DISPLACEMENT_STD": disp_stats.std()},
        cell_data={"YOUNG_MODULUS_MEAN": E_stats.mean(),
                   "YOUNG_MODULUS_STD": E_stats.std()},
        title=title,
    )
    return True
