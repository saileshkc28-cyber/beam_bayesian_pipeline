import numpy as np


class FieldStats:
    """Welford running mean/std. Never solve at the mean alpha: u(E[a]) != E[u(a)]."""

    def __init__(self):
        self.n, self.mu, self.m2 = 0, None, None

    def update(self, field):
        field = np.asarray(field, float)
        if self.mu is None:
            self.mu, self.m2 = np.zeros_like(field), np.zeros_like(field)
        self.n += 1
        delta = field - self.mu
        self.mu += delta / self.n
        self.m2 += delta * (field - self.mu)

    def mean(self):
        return self.mu

    def std(self):
        return np.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else np.zeros_like(self.m2)


def write_vtk(path, model, disp_stats, E_stats):
    """VtkOutputProcess can only emit registered variables; these names are not."""
    nodes, cells = model.nodes, model.cells
    with open(path, "w") as f:
        f.write(f"# vtk DataFile Version 4.2\nposterior statistics\nASCII\n"
                f"DATASET UNSTRUCTURED_GRID\nPOINTS {len(nodes)} double\n")
        f.writelines(f"{p[0]:.10e} {p[1]:.10e} {p[2]:.10e}\n" for p in nodes)
        f.write(f"\nCELLS {len(cells)} {4 * len(cells)}\n")
        f.writelines(f"3 {c[0]} {c[1]} {c[2]}\n" for c in cells)
        f.write(f"\nCELL_TYPES {len(cells)}\n" + "5\n" * len(cells))

        f.write(f"\nPOINT_DATA {len(nodes)}\n")
        for name, arr in (("DISPLACEMENT_MEAN", disp_stats.mean()),
                          ("DISPLACEMENT_STD", disp_stats.std())):
            f.write(f"VECTORS {name} double\n")
            f.writelines(f"{v[0]:.10e} {v[1]:.10e} {v[2]:.10e}\n" for v in arr)

        f.write(f"\nCELL_DATA {len(cells)}\n")
        for name, arr in (("YOUNG_MODULUS_MEAN", E_stats.mean()),
                          ("YOUNG_MODULUS_STD", E_stats.std())):
            f.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n")
            f.writelines(f"{v:.10e}\n" for v in arr)
