from dolfin import *
import numpy as np


def integrate_force_strong(u, p, nu, mesh, dmeas, did):
    I = Identity(mesh.geometry().dim())
    n = FacetNormal(mesh)

    D = 0.5 * (grad(u) + grad(u).T)  # or D=sym(grad(v))
    T = -p * I + 2 * nu * D
    force = T * n
    fl = [assemble(force[i] * dmeas(did)) for i in range(mesh.geometry().dim())]
    return np.array(fl)


def calc_force_coeffs_turek(u_h, p_h, mu, n, ds, resdict, len_scale=1.0):
    """
    assume that resdict is a defaultdict(list)
    this convention for the calculation of lift and drag coefficients is the 
    one used in the turek CFD benchmark
    """
    n1 = -n
    u_t = inner(as_vector((n1[1], -n1[0])), u_h)
    drag = assemble(
        2 / len_scale * (mu * inner(grad(u_t), n1) * n1[1] - p_h * n1[0]) * ds
    )
    lift = assemble(
        -2 / len_scale * (mu * inner(grad(u_t), n1) * n1[0] + p_h * n1[1]) * ds
    )
    resdict["c_ds"].append(drag)
    resdict["c_ls"].append(lift)
    return
