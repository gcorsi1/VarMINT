"""
This demo solves the 2D VIV FSI problem introduced in the article of Peric 
and Dettmer on CMAME (2006).  
"""
from VarMINT import *
from VarMINTpostproc import calc_force_coeffs
import math
import numpy as np
from collections import defaultdict

####### Parameters #######

# Arguments are parsed from the command line, with some hard-coded defaults
# here.  See help messages for descriptions of parameters.

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--Nel", dest="Nel", default=16, help="Number of elements in each direction."
)
parser.add_argument("--Re", dest="Re", default=100.0, help="Reynolds number.")
parser.add_argument("--k", dest="k", default=1, help="Polynomial degree.")
parser.add_argument(
    "--T", dest="T", default=1.0, help="Length of time interval to consider."
)
parser.add_argument(
    "--rhoinf",
    dest="rhoinf",
    default=1.0,
    help="The rho_inf parameter in the generalized alpha method.",
)


args = parser.parse_args()
Nel = int(args.Nel)
Re = Constant(float(args.Re))
k = int(args.k)
T = float(args.T)
rhoinf = float(args.rhoinf)

# store results
results = defaultdict(list)
####### Analysis #######

# Avoid overkill automatically-determined quadrature degree:
QUAD_DEG = 2 * k
dx = dx(metadata={"quadrature_degree": QUAD_DEG})

# Domain:
mesh = Mesh(MPI.comm_world)
with XDMFFile("bfile_VIV.xdmf") as file:
    file.read(mesh)

mtot_ = MPI.sum(MPI.comm_world, mesh.num_cells())
print(f"Read mesh with {mtot_} cells.")

# Read boundary data
mvc_boundaries = MeshValueCollection("size_t", mesh, mesh.topology().dim() - 1)
with XDMFFile("domains_VIV.xdmf") as file:
    file.read(mvc_boundaries)
boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc_boundaries)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
n = FacetNormal(mesh)

# Mixed velocity--pressure space:
V = equalOrderSpace(mesh, k=k)

# Solution and test functions:
up = Function(V)
u, p = split(up)
vq = TestFunction(V)
v, q = split(vq)

# Determine physical parameters from Reynolds number:
rho = Constant(1.0)
mu = Constant(1.0 / Re)
nu = mu / rho

# Constants of the genralized alpha method
alpha_mf = 0.5 * (3.0 - rhoinf) / (1.0 + rhoinf)
alpha_ff = 1.0 / (1.0 + rhoinf)
gamma_f = 0.5 + alpha_mf - alpha_ff

# Generalized alpha time integration:
N_STEPS = Nel  # Space--time quasi-uniformity
Dt = Constant(T / N_STEPS)
up_old = Function(V)
upt, upt_old = (Function(V), Function(V))
ut, _ = split(upt)
ut_old, _ = split(upt_old)
u_old, _ = split(up_old)
u_alpha = alpha_ff * u + (1.0 - alpha_ff) * u_old
u_t_alpha = (1.0 - alpha_mf / gamma_f) * ut_old + alpha_mf / Dt / gamma_f * (u - u_old)

# Initial condition for the Taylor--Green vortex
x = SpatialCoordinate(mesh)
u_IC = as_vector((Constant(0.0), Constant(0.0)))


class BoundaryFunction(UserExpression):
    def __init__(self, t, **kwargs):
        super().__init__(**kwargs)
        self.t = t

    def eval(self, values, x):
        U = 1.5 * sin(pi * self.t / 8)
        values[0] = 4 * U * x[1] * (0.41 - x[1]) / pow(0.41, 2)
        values[1] = 0

    def value_shape(self):
        return (2,)


# Project the initial condition:
up_old.assign(project(as_vector((u_IC[0], u_IC[1], Constant(0.0))), V))

# Init all to zero
for ui in (up, upt, upt_old):
    ui.assign(up_old)

# Mesh motion stuff
VM = V.sub(0).collapse()
s = TrialFunction(VM)
z = TestFunction(VM)
Sm = Function(VM)
Sm0 = Function(VM)
# v_mesh = Function(VM)
omega = np.array([0.0, 0.0, 0.025])
vte_x = Expression("Ux - omega_z*x[1]", Ux=0.0, omega_y=0.0, omega_z=0.0, degree=2)
vte_y = Expression("Uy + omega_z*x[0]", Uy=0.0, omega_x=0.0, omega_z=0.0, degree=2)
v_mesh = as_vector((vte_x, vte_y))

# Project the initial condition:
up_old.assign(project(as_vector((u_IC[0], u_IC[1], Constant(0.0))), V))

# Same-velocity predictor:
up.assign(up_old)

# boundary markers
ball = 1
inflow = 2
outflow = 3
bottom = 4
top = 5

# Define boundary conditions
U_inlet = BoundaryFunction(0.0)
bc_inlet = DirichletBC(V.sub(0), U_inlet, boundaries, inflow)
bc_bottom = DirichletBC(V.sub(0), Constant((0.0, 0.0)), boundaries, bottom)
bc_top = DirichletBC(V.sub(0), Constant((0.0, 0.0)), boundaries, top)
bc_obstacle = DirichletBC(V.sub(0), Constant((0.0, 0.0)), boundaries, ball)
bc_outlet = DirichletBC(V.sub(1), Constant(0.0), boundaries, outflow)
bcs = [bc_inlet, bc_top, bc_bottom, bc_obstacle, bc_outlet]

# Define boundary conditions for mesh motion
bc_m_inlet = DirichletBC(VM, Constant((0.0, 0.0)), boundaries, inflow)
bc_m_bottom = DirichletBC(VM, Constant((0.0, 0.0)), boundaries, bottom)
bc_m_top = DirichletBC(VM, Constant((0.0, 0.0)), boundaries, top)
bc_m_obstacle = DirichletBC(VM, v_mesh, boundaries, ball)
bc_m_outlet = DirichletBC(VM, Constant((0.0, 0.0)), boundaries, outflow)
bcs_m = [bc_m_inlet, bc_m_top, bc_m_bottom, bc_m_obstacle, bc_m_outlet]


# Weak problem residual; note use of midpoint velocity:
F = interiorResidual(
    u_alpha,
    p,
    v,
    q,
    rho,
    mu,
    mesh,
    u_t=u_t_alpha,
    Dt=Dt,
    C_I=Constant(6.0 * (k**4)),
    dx=dx,
    u_mesh=v_mesh,
)

a_m = inner(grad(s), grad(z)) * dx
f_m = inner(Constant((0.0, 0.0)), z) * dx

t = 0.0
results["ts"].append(t)

# Time stepping loop:
with XDMFFile("solu.xdmf") as fileu, XDMFFile("solp.xdmf") as filep:

    fileu.parameters.update(
        {"functions_share_mesh": True, "rewrite_function_mesh": True}
    )

    filep.parameters.update(
        {"functions_share_mesh": True, "rewrite_function_mesh": True}
    )

    for step in range(0, N_STEPS):
        t += float(Dt)
        print("======= Time step " + str(step + 1) + "/" + str(N_STEPS) + " =======")

        # Update dirichlet boundary condition:
        vte_x.omega_z = omega[2]
        vte_y.omega_z = omega[2]
        U_inlet.t = t

        Am, Lm = assemble_system(a_m, f_m, bcs_m)
        Sm0.vector().zero()
        Sm0.vector().axpy(1.0, Sm.vector())
        solve(a_m == f_m, Sm, bcs=bcs_m)
        print("CALLING ALE MOVE.")
        # Move mesh
        ALE.move(mesh, Sm)

        solve(F == 0, up, bcs=bcs)
        up_old.assign(up)

        uf, pf = up.split(deepcopy=True)
        uf.rename("Velocity", "Velocity")
        pf.rename("Pressure", "Pressure")
        fileu.write(uf, step)
        filep.write(pf, step)

        results["ts"].append(t)
        calc_force_coeffs(u, p, mu, n, ds(1), results)
    np.savez(
        "resultsALE",
        CD=np.array(results["c_ds"]),
        CL=np.array(results["c_ls"]),
        t=np.array(results["ts"]),
    )

print("End of time loop.")
