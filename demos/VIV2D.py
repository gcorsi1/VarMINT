"""
This demo solves the 2D VIV FSI problem introduced in the article of Peric 
and Dettmer on CMAME (2006).  
"""
from VarMINT import *
from VarMINTpostproc import integrate_force
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
Dt = float(T / N_STEPS)
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
vbc = Expression(
    ("t < 4.0 ? U*(1-std::cos(3.14142/2.0*t))/2.0 : U", "0.0"), U=1.0, t=0.0, degree=2
)  # ------INLET WITH STARTUP PROFILE

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
v_mesh = Function(VM)
v_mesh_old = Function(VM)
# rigid velocity expression
vte_x = Expression("Ux - omega_z*x[1]", Ux=0.0, omega_y=0.0, omega_z=0.0, degree=2)
vte_y = Expression("Uy + omega_z*x[0]", Uy=0.0, omega_x=0.0, omega_z=0.0, degree=2)
v_mesh_exp = as_vector((vte_x, vte_y))

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
bc_inlet = DirichletBC(V.sub(0), vbc, boundaries, inflow)
bc_bottom = DirichletBC(V.sub(0), Constant((0.0, 0.0)), boundaries, bottom)
bc_top = DirichletBC(V.sub(0), Constant((0.0, 0.0)), boundaries, top)
bc_outlet = DirichletBC(V.sub(1), Constant(0.0), boundaries, outflow)
bcs = [bc_inlet, bc_top, bc_bottom, bc_outlet]

# Define boundary conditions for mesh motion
bc_m_inlet = DirichletBC(VM, Constant((0.0, 0.0)), boundaries, inflow)
bc_m_bottom = DirichletBC(VM, Constant((0.0, 0.0)), boundaries, bottom)
bc_m_top = DirichletBC(VM, Constant((0.0, 0.0)), boundaries, top)
bc_m_outlet = DirichletBC(VM, Constant((0.0, 0.0)), boundaries, outflow)
bcs_m = [bc_m_inlet, bc_m_top, bc_m_bottom, bc_m_outlet]

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
    C_I=Constant(6.0 * (k ** 4)),
    dx=dx,
    u_mesh=v_mesh,
)

a_m = inner(grad(s), grad(z)) * dx
f_m = inner(Constant((0.0, 0.0)), z) * dx

# SOLID PROBLEM
N = 2
y, ydot, yddot = [np.zeros(N) for _ in range(3)]
y_old, ydot_old, yddot_old = [np.zeros(N) for _ in range(3)]

# for now, rhoinf is set to be the same as that of the fluid
rhoinf_r = rhoinf
alpha_fr = 1.0 / (1.0 + rhoinf_r)
alpha_mr = (2.0 - rhoinf_r) / (1.0 + rhoinf_r)
beta_r = 0.25 * (1 + alpha_mr - alpha_fr) ** 2
gamma_r = 0.5 + alpha_mr - alpha_fr


def Mm():
    return np.array([[1.0, 0.0], [0.0, 1.0]])


def Km():
    return np.array([[0.215 * 2 * pi, 0.0], [0.0, 0.215 * 2 * pi]])


def Cm():
    return np.array([[0.01, 0.0], [0.0, 0.01]])


def restoring_force(disp, velocity):
    return Cm() @ velocity + Km() @ disp


def solid_alpha(
    alpha_fr, alpha_mr, beta_r, gamma_r, Dt, ydot, y_old, ydot_old, yddot_old
):
    """
    extrapolated quantities, from old time values, for generalized alpha
    method of time integration
    """
    y_alpha = (
        y_old
        + Dt * alpha_fr * (gamma_r - beta_r) / gamma_r * ydot_old
        + Dt ** 2 * alpha_fr * (gamma_r - 2 * beta_r) / 2.0 / gamma_r * yddot_old
        + Dt * alpha_fr * beta_r / gamma_r * ydot
    )
    ydot_alpha = (1 - alpha_fr) * ydot_old + alpha_fr * ydot
    yddot_alpha = (1.0 - alpha_mr) / gamma_r * yddot_old + alpha_mr / Dt / gamma_r * (
        ydot - ydot_old
    )
    return y_alpha, ydot_alpha, yddot_alpha


def solid_step_nonlinear(
    alpha_fr,
    alpha_mr,
    beta_r,
    gamma_r,
    Dt,
    Mm,
    Cm,
    Km,
    Fext,
    ydot,
    y_old,
    ydot_old,
    yddot_old,
    logs=False,
):
    """
    advance in time following the generalized alpha method, as written by Peric
    results in the new value of solid velocity. Assume the general case when
    the solid ODE has a nonlinear component, depending on velocity and
    displacements (linear in the accelerations)
    """
    # predictor step
    y_alpha = (
        y_old
        + Dt * alpha_fr * (gamma_r - beta_r) / gamma_r * ydot_old
        + Dt ** 2 * alpha_fr * (gamma_r - 2 * beta_r) / 2.0 / gamma_r * yddot_old
        + Dt * alpha_fr * beta_r / gamma_r * ydot
    )
    ydot_alpha = (1 - alpha_fr) * ydot_old + alpha_fr * ydot
    yddot_alpha = (1.0 - alpha_mr) / gamma_r * yddot_old + alpha_mr / Dt / gamma_r * (
        ydot - ydot_old
    )
    # residual of the nonlinear ODE
    eps = Fext - restoring_force(y_alpha, ydot_alpha) - Mm() @ yddot_alpha

    alphatol = 1e-6
    it = 0

    # store ydot now, return a copy since the code might iterate if the
    # block iterative approach is used
    ynew = ydot.copy()
    while np.linalg.norm(eps) > alphatol:
        it += 1
        mat_new = (
            alpha_mr / Dt / gamma_r * Mm()
            + alpha_fr * Cm(y_alpha, ydot_alpha)
            + Dt * beta_r * alpha_fr / gamma_r * Km(y_alpha, ydot_alpha)
        )
        delta_v = np.linalg.inv(mat_new) @ eps

        # update step
        ynew += delta_v
        y_alpha += Dt * beta_r * alpha_fr / gamma_r * delta_v
        ydot_alpha += alpha_fr * delta_v
        yddot_alpha += alpha_mr / Dt / gamma_r * delta_v
        np.copyto(eps, Fext - restoring_force(y_alpha, ydot_alpha) - Mm() @ yddot_alpha)

    if logs:
        if MPI.rank(MPI.comm_world) == 0:
            print(
                f"Generalized-alpha method: solid subproblem converged in {it} iterations."
            )
    return ynew


def solid_step_linear(
    alpha_fr, alpha_mr, beta_r, gamma_r, Dt, M, C, K, Fext, y_old, ydot_old, yddot_old
):
    """
    advance in time following the generalized alpha method, as written by Peric
    results in the new value of solid velocity. M, C, K are assumed to be
    constant here
    """
    resd = (
        2 * alpha_mr * M @ (Dt * yddot_old + ydot_old)
        - 2 * Dt * (M @ yddot_old + gamma_r * (-Fext + K @ y_old + C @ ydot_old))
        + Dt
        * alpha_fr
        * (
            2 * Dt * K @ (Dt * yddot_old + ydot_old) * beta_r
            - gamma_r
            * (Dt ** 2 * K @ yddot_old - 2 * C @ ydot_old + 2 * Dt * K @ ydot_old)
        )
    )
    mat = 2 * (alpha_mr * M + Dt * alpha_fr * (Dt * beta_r * K + gamma_r * C))
    invmat = 1.0 / mat if len(mat.shape) == 1 else np.linalg.inv(mat)
    ydot_new = invmat @ resd
    # recast to np.array if needed (ie if the ode is 1d and result is a float)
    return np.array(ydot_new)


def solid_corrector(beta_r, gamma_r, Dt, ydot, y_old, ydot_old, yddot_old):
    """
    given value of solid velocity at new time, calculate the acceleration and
    displacement
    """
    y_new = (
        y_old
        + Dt * (gamma_r - beta_r) / gamma_r * ydot_old
        + Dt ** 2 * (gamma_r - 2 * beta_r) / 2 / gamma_r * yddot_old
        + Dt * beta_r / gamma_r * ydot
    )
    yddot_new = -(1 - gamma_r) / gamma_r * yddot_old + 1.0 / Dt / gamma_r * (
        ydot - ydot_old
    )
    return y_new, yddot_new


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

        np.copyto(y_old, y)
        np.copyto(ydot_old, ydot)
        np.copyto(yddot_old, yddot)
        Fext = -1.0 * integrate_force(u, p, mu, mesh, ds, 1)
        Fext[0] = 0.0
        if t < 0.5:
            Fext[1] = 0.0
        ydot_new = solid_step_nonlinear(
            alpha_fr,
            alpha_mr,
            beta_r,
            gamma_r,
            Dt,
            Mm,
            Cm,
            Km,
            Fext,
            y_old,
            ydot_old,
            yddot_old,
        )
        np.copyto(ydot, ydot_new)
        vte_y.Uy = ydot[1]
        bc_m_obstacle = DirichletBC(VM, v_mesh_exp, boundaries, ball)

        # lift velocity at interface
        # Am, Lm = assemble_system(a_m, f_m, bcs_m)
        v_mesh_old.vector().zero()
        v_mesh_old.vector().axpy(1.0, v_mesh.vector())
        solve(a_m == f_m, v_mesh, bcs=bcs_m + [bc_m_obstacle])
        print("CALLING ALE MOVE.")
        Sm.vector().zero()
        # Trapezoidal rule
        Sm.vector().axpy(Dt / 2.0, v_mesh.vector())
        Sm.vector().axpy(Dt / 2.0, v_mesh_old.vector())
        # Move mesh
        ALE.move(mesh, Sm)

        # Now advance fluid in time
        # Update dirichlet boundary condition:
        vbc.t = t
        bc_obstacle = DirichletBC(V.sub(0), v_mesh, boundaries, ball)

        solve(F == 0, up, bcs=bcs + [bc_obstacle])
        # Corrector step
        upt.assign(
            1.0 / float(Dt) / gamma_f * (up - up_old)
            - (1.0 - gamma_f) / gamma_f * upt_old
        )
        up_old.assign(up)
        y_new, yddot_new = solid_corrector(
            beta_r, gamma_r, Dt, ydot, y_old, ydot_old, yddot_old
        )
        np.copyto(y, y_new)
        np.copyto(yddot, yddot_new)

        uf, pf = up.split(deepcopy=True)
        uf.rename("Velocity", "Velocity")
        pf.rename("Pressure", "Pressure")
        fileu.write(uf, step)
        filep.write(pf, step)

        results["ts"].append(t)
        results["disp_y"].append(y[1])

print("End of time loop.")
