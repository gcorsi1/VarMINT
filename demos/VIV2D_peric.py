"""
This demo solves the 2D VIV FSI problem introduced in the article of Peric 
and Dettmer on CMAME (2006).  
"""
import argparse
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time

from VarMINT import *
from VarMINTpostproc import integrate_force_strong

matplotlib.use("Agg")

# Suppress excessive output in parallel:
parameters["std_out_all_processes"] = False

# results postprocess function
def postprocess(results_dict):
    data_preproc = pd.DataFrame.from_dict(results_dict)
    data_preproc.to_csv("./raw_data.csv")
    data_preproc["disp_x"] = data_preproc["disp_x"].div(D)
    data_preproc["disp_y"] = data_preproc["disp_y"].div(D)

    fade_ = 4  # DOWNSAMPLE FACTOR FOR THE PLOTS
    data_preproc = data_preproc.iloc[
        1::fade_, :
    ]  # only get some of the rows in the plot
    # For plots, might remove initial seconds where pressure wave gives spurious
    # very high values
    # data_preproc = data_preproc[data_preproc.ts > 2]
    g = sns.lineplot(
        x="ts",
        y="value",
        hue="variable",
        hue_order=["CL", "CD"],
        marker="o",
        data=pd.melt(data_preproc, ["ts"]),
    )
    sns.despine()
    plt.xlabel(r"$t [s]$")
    plt.ylabel("")
    plt.legend(
        title="Coefficient",
        bbox_to_anchor=(0.9, 0.9),
        loc=2,
        labels=[r"$C_L$", r"$C_D$"],
    )
    plt.savefig("coefficients.pdf")
    plt.close()

    # now plot displacement over time
    g = sns.lineplot(
        x="ts",
        y="value",
        hue="variable",
        marker="o",
        hue_order=["disp_x", "disp_y"],
        data=pd.melt(data_preproc, ["ts"]),
    )
    sns.despine()
    plt.xlabel(r"$t [s]$")
    plt.ylabel("")
    plt.legend(
        title="Displacement",
        bbox_to_anchor=(0.9, 0.9),
        loc=2,
        labels=[r"$disp_x/D$", r"$disp_y/D$"],
    )
    plt.savefig("disp_xy.pdf")
    plt.close()


####### Parameters #######

# Arguments are parsed from the command line, with some hard-coded defaults
# here.  See help messages for descriptions of parameters.


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
D = 1.0

# store results
results = defaultdict(list)
####### Analysis #######

# Avoid overkill automatically-determined quadrature degree:
QUAD_DEG = 2 * k
dx = dx(metadata={"quadrature_degree": QUAD_DEG})

# Domain:
mesh = Mesh(MPI.comm_world)
with XDMFFile("bfile_VIV_square.xdmf") as file:
    file.read(mesh)

mtot_ = MPI.sum(MPI.comm_world, mesh.num_cells())
print(f"Read mesh with {mtot_} cells.")

# Read boundary data
mvc_boundaries = MeshValueCollection("size_t", mesh, mesh.topology().dim() - 1)
with XDMFFile("domains_VIV_square.xdmf") as file:
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
mu = Constant(1.0e-2)
Uinf = Constant(mu * Re / D)
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
# Always use U value of 1.0 at infinity, adjust viscosity to get the required
# Reynolds number
vbc = Expression(
    ("t < 2.0 ? U*(1-std::cos(3.14142/2.0*t))/2.0 : U", "0.0"), U=Uinf, t=0.0, degree=2
)  # Inlet with startup profile

# Project the initial condition:
up_old.assign(project(as_vector((u_IC[0], u_IC[1], Constant(0.0))), V))

# Init all to zero
for ui in (up, upt, upt_old):
    ui.assign(up_old)

# Mesh motion stuff
VM = V.sub(0).collapse()
QM = V.sub(1).collapse()
s = TrialFunction(VM)
z = TestFunction(VM)
Sm = Function(VM)
Sm0 = Function(VM)
v_mesh = Function(VM)
v_mesh_old = Function(VM)
v_mesh_alpha = Function(VM)
# Rigid velocity expression
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
    C_I=Constant(6.0 * (k**4)),
    dx=dx,
    u_mesh=v_mesh_alpha,
)

a_m = inner(grad(s), grad(z)) * dx
f_m = inner(Constant((0.0, 0.0)), z) * dx

# Solid problem
N = 2
y, ydot, yddot = [np.zeros(N) for _ in range(3)]
y_old, ydot_old, yddot_old = [np.zeros(N) for _ in range(3)]
ydot_iter, ydot_iter_old = [np.zeros(N) for _ in range(2)]
residual, residual_old = [np.zeros(N) for _ in range(2)]

# For now, rhoinf is set to be the same as that of the fluid
rhoinf_r = rhoinf
alpha_fr = 1.0 / (1.0 + rhoinf_r)
alpha_mr = (2.0 - rhoinf_r) / (1.0 + rhoinf_r)
beta_r = 0.25 * (1 + alpha_mr - alpha_fr) ** 2
gamma_r = 0.5 + alpha_mr - alpha_fr
k = 12.0 * np.pi**2 / 25.0  # [N/m]
c = 0.0  # [Kg/s]
m = 3.0  # [Kg]


def Mm():
    return np.array([[m, 0.0], [0.0, m]])


def Km(y, ydot):
    return np.array([[k, 0.0], [0.0, k]])


def Cm(y, ydot):
    return np.array([[c, 0.0], [0.0, c]])


def restoring_force(K, C, disp, velocity):
    return C @ velocity + K @ disp


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
        + Dt**2 * alpha_fr * (gamma_r - 2 * beta_r) / 2.0 / gamma_r * yddot_old
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
    # Predictor step, notice ydot_old appears twice, here we simply start from
    # the known value at old time step
    y_alpha, ydot_alpha, yddot_alpha = solid_alpha(
        alpha_fr, alpha_mr, beta_r, gamma_r, Dt, ydot_old, y_old, ydot_old, yddot_old
    )
    # Residual of the nonlinear ODE
    MM = Mm()
    KM = Km(y_alpha, ydot_alpha)
    CM = Cm(y_alpha, ydot_alpha)
    eps = Fext - restoring_force(KM, CM, y_alpha, ydot_alpha) - MM @ yddot_alpha

    alphatol = 1e-10
    it = 0

    # Store ydot now, return a copy since the code might iterate if the
    # block iterative approach is used
    ynew = ydot_old.copy()
    while np.linalg.norm(eps) > alphatol:
        it += 1
        KM = Km(y_alpha, ydot_alpha)
        CM = Cm(y_alpha, ydot_alpha)
        mat_new = (
            alpha_mr / Dt / gamma_r * MM
            + alpha_fr * CM
            + Dt * beta_r * alpha_fr / gamma_r * KM
        )
        delta_v = np.linalg.inv(mat_new) @ eps

        # Update step
        ynew += delta_v
        y_alpha += Dt * beta_r * alpha_fr / gamma_r * delta_v
        ydot_alpha += alpha_fr * delta_v
        yddot_alpha += alpha_mr / Dt / gamma_r * delta_v
        np.copyto(
            eps, Fext - restoring_force(KM, CM, y_alpha, ydot_alpha) - MM @ yddot_alpha
        )

    if logs:
        if MPI.rank(MPI.comm_world) == 0:
            print(
                f"Generalized-alpha method: solid subproblem converged in {it} iterations. "
                f"Norm of residual {np.linalg.norm(eps)}."
            )
    return ynew


def solid_step_linear(
    alpha_fr, alpha_mr, beta_r, gamma_r, Dt, M, C, K, Fext, y_old, ydot_old, yddot_old
):
    """
    Advance in time following the generalized alpha method, as written by Peric
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
            * (Dt**2 * K @ yddot_old - 2 * C @ ydot_old + 2 * Dt * K @ ydot_old)
        )
    )
    mat = 2 * (alpha_mr * M + Dt * alpha_fr * (Dt * beta_r * K + gamma_r * C))
    invmat = 1.0 / mat if len(mat.shape) == 1 else np.linalg.inv(mat)
    ydot_new = invmat @ resd
    # Recast to np.array if needed (ie if the ode is 1d and result is a float)
    return np.array(ydot_new)


def solid_corrector(beta_r, gamma_r, Dt, ydot, y_old, ydot_old, yddot_old):
    """
    Given value of solid velocity at new time, calculate the acceleration and
    displacement
    """
    y_new = (
        y_old
        + Dt * (gamma_r - beta_r) / gamma_r * ydot_old
        + Dt**2 * (gamma_r - 2 * beta_r) / 2 / gamma_r * yddot_old
        + Dt * beta_r / gamma_r * ydot
    )
    yddot_new = -(1 - gamma_r) / gamma_r * yddot_old + 1.0 / Dt / gamma_r * (
        ydot - ydot_old
    )
    return y_new, yddot_new


def fsi_postprocessing(
    ydot_iter,
    ydot_iter_old,
    ydot_old,
    fsi_resd,
    residual,
    iter_fsi,
    residual_old,
    omega_old,
):
    """
    Calculate the normalized residual for the fsi iteration, and relax the
    iteration result with the Aitken acceleration method
    """
    if MPI.rank(MPI.comm_world) == 0:
        print("YDOT ITER, YDOT ITER OLD", ydot_iter, ydot_iter_old)
    np.copyto(residual_old, residual)
    residual_new = ydot_iter - ydot_iter_old
    if MPI.rank(MPI.comm_world) == 0:
        print("RESIDUAL, RESIDUAL OLD", residual_new, residual_old)
    res_ = np.linalg.norm(residual_new) / (np.linalg.norm(ydot_old) + 1.0e-8)
    if MPI.rank(MPI.comm_world) == 0:
        print(f"RESIDUAL NORM IS {res_}")
    np.copyto(fsi_resd, res_)
    np.copyto(residual, residual_new)
    omega_new = (
        -1.0
        * omega_old[0]
        * np.dot(residual_old, (residual - residual_old))
        / (np.linalg.norm(residual - residual_old) ** 2 + 1.0e-24)
    )
    if MPI.rank(MPI.comm_world) == 0:
        print(f"OMEGA NEW VALUE RAW: {omega_new}")
    omega_new = max(min(omega_new, 0.99), 0.2)
    ydot_new_post = omega_new * ydot_iter + (1.0 - omega_new) * ydot_iter_old
    if MPI.rank(MPI.comm_world) == 0:
        print(f"ITER: {iter_fsi}. OMEGA VALUE: {omega_new}. SOLID VELOCITY IS: {ydot}")
    return ydot_new_post, np.array(omega_new)


# define the functions that will be used to integrate the forces on the
# cylinder variationally, these will be equal to a cartesian base vector on
# the surface of the cylinder, and 0 on the other Dirichlet boundaries.
# Most of the stuff needed was already defined when writing down the mesh
# motion problem
bc_m_obstacle_e0 = DirichletBC(VM, Constant((1.0, 0.0)), boundaries, ball)
bc_m_obstacle_e1 = DirichletBC(VM, Constant((0.0, 1.0)), boundaries, ball)
bcs_f = [bc_m_inlet, bc_m_bottom, bc_m_top]
eis = [Function(VM) for _ in range(mesh.geometry().dim())]

# solve an auxiliary laplacian problem in order to calculate the e_i
# and define a zero function to be used for the pressure test function
for eii, bci in zip(eis, [bc_m_obstacle_e0, bc_m_obstacle_e1]):
    solve(a_m == f_m, eii, bcs=bcs_f + [bci])
q0 = Function(QM)
q0.assign(project(Constant(0.0), QM))

# Calculate force variationally
F0 = interiorResidual(
    u,
    p,
    eis[0],
    q0,
    rho,
    mu,
    mesh,
    u_t=ut,
    Dt=Dt,
    C_I=Constant(6.0 * (k**4)),
    dx=dx,
    u_mesh=v_mesh_alpha,
)

# Calculate force variationally
F1 = interiorResidual(
    u,
    p,
    eis[1],
    q0,
    rho,
    mu,
    mesh,
    u_t=ut,
    Dt=Dt,
    C_I=Constant(6.0 * (k**4)),
    dx=dx,
    u_mesh=v_mesh_alpha,
)


t = 0.0
results["ts"].append(t)
results["disp_y"].append(0.0)
results["disp_x"].append(0.0)
results["CD"].append(-2.0 * assemble(F0))
results["CL"].append(-2.0 * assemble(F1))
results["freq"].append(0.0)
max_iter_fsi = 50
fsi_tol = 1e-4

# Time stepping loop:
start = time.time()
with XDMFFile("solup.xdmf") as fileup:

    fileup.parameters.update(
        {"functions_share_mesh": True, "rewrite_function_mesh": True}
    )

    for step in range(0, N_STEPS):
        t += float(Dt)
        if MPI.rank(MPI.comm_world) == 0:
            print(
                "======= Time step " + str(step + 1) + "/" + str(N_STEPS) + " ======="
            )
        iter_fsi = 0
        fsi_resd = np.array([1e8])
        omega_old = np.array([0.2])

        # Update dirichlet boundary condition:
        vbc.t = t

        # Mesh velocity and motion predictor
        bc_m_obstacle = DirichletBC(VM, v_mesh_exp, boundaries, ball)
        v_mesh_old.vector().zero()
        v_mesh_old.vector().axpy(1.0, v_mesh.vector())
        solve(a_m == f_m, v_mesh_old, bcs=bcs_m + [bc_m_obstacle])
        v_mesh_alpha.vector().zero()
        v_mesh_alpha.vector().axpy(1.0, v_mesh_old.vector())
        Sm0.vector().zero()
        Sm0.vector().axpy(alpha_ff * Dt, v_mesh_old.vector())
        if MPI.rank(MPI.comm_world) == 0:
            print("CALLING ALE MOVE.")
        ALE.move(mesh, Sm0)

        while iter_fsi < max_iter_fsi and fsi_resd[0] > fsi_tol:

            Fext = -1.0 * integrate_force_strong(u, p, mu, mesh, ds, 1)
            # Actually use the variational calculation
            fx0, fx1 = assemble(F0), assemble(F1)
            if MPI.rank(MPI.comm_world) == 0:
                print(f"{fx0=}, {fx1=}, {Fext=}")
            Fext[0] = 0.0
            # Wait for wake to develop
            if t < 2.5:
                np.copyto(Fext, np.zeros(2))
            else:
                np.copyto(Fext, np.array([-1.0 * fx0, -1.0 * fx1]))

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
                logs=True,
            )
            np.copyto(ydot_iter_old, ydot_iter)
            np.copyto(ydot_iter, ydot_new)

            ydot_new_post, omega_new = fsi_postprocessing(
                ydot_iter,
                ydot_iter_old,
                ydot_old,
                fsi_resd,
                residual,
                iter_fsi,
                residual_old,
                omega_old,
            )

            np.copyto(ydot_iter, ydot_new_post)
            np.copyto(ydot, ydot_new_post)
            # Update mesh velocity for next time-step
            vte_x.Ux = ydot[0]
            vte_y.Uy = ydot[1]
            np.copyto(omega_old, omega_new)

            # Now advance fluid in time
            _, ydot_alpha, _ = solid_alpha(
                alpha_fr,
                alpha_mr,
                beta_r,
                gamma_r,
                Dt,
                ydot,
                y_old,
                ydot_old,
                yddot_old,
            )
            bc_obstacle = DirichletBC(V.sub(0), ydot_alpha, boundaries, ball)
            solve(F == 0, up, bcs=bcs + [bc_obstacle])

            # Lift mesh velocity at interface
            bc_m_obstacle = DirichletBC(VM, v_mesh_exp, boundaries, ball)
            solve(a_m == f_m, v_mesh, bcs=bcs_m + [bc_m_obstacle])
            v_mesh_alpha.vector().zero()
            v_mesh_alpha.vector().axpy(1.0 - alpha_ff, v_mesh_old.vector())
            v_mesh_alpha.vector().axpy(alpha_ff, v_mesh.vector())
            Sm.vector().zero()
            Sm.vector().axpy(alpha_ff * Dt * gamma_f, v_mesh.vector())
            Sm.vector().axpy(alpha_ff * Dt * (1.0 - gamma_f), v_mesh_old.vector())
            Sm.vector().axpy(-1.0, Sm0.vector())
            if MPI.rank(MPI.comm_world) == 0:
                print("CALLING ALE MOVE.")
            ALE.move(mesh, Sm)
            Sm0.vector().zero()
            Sm0.vector().axpy(1.0, Sm.vector())

            iter_fsi += 1

        if MPI.rank(MPI.comm_world) == 0:
            print(f"Converged in {iter_fsi} FSI iterations.")
        # Corrector step for fluid
        upt.assign(
            1.0 / float(Dt) / gamma_f * (up - up_old)
            - (1.0 - gamma_f) / gamma_f * upt_old
        )
        up_old.assign(up)
        upt_old.assign(upt)
        # Corrector step for solid
        y_new, yddot_new = solid_corrector(
            beta_r, gamma_r, Dt, ydot, y_old, ydot_old, yddot_old
        )
        np.copyto(y_old, y_new)
        np.copyto(ydot_old, ydot)
        np.copyto(yddot_old, yddot_new)

        results["ts"].append(t)
        results["disp_x"].append(y_old[0])
        results["disp_y"].append(y_old[1])
        results["CD"].append(-2.0 * assemble(F0))
        results["CL"].append(-2.0 * assemble(F1))

        # calculate the main frequency of oscillation
        Ntot = len(results["CL"])
        ws = np.fft.fft(results["CL"])
        Ttot = results["ts"][-1]
        dfs = 1.0 / Ttot
        freqs = np.fft.fftfreq(Ntot) * Ntot * dfs
        idx = np.argmax(np.abs(ws))
        maxf = np.abs(freqs[idx])
        if MPI.rank(MPI.comm_world) == 0:
            print(f"main frequency sampled from data is: {maxf:.4f}")
        results["freq"].append(maxf)

        if not step % 100:
            postprocess(results)
            # Save some results
            uf, pf = up.split(deepcopy=True)
            uf.rename("Velocity", "Velocity")
            pf.rename("Pressure", "Pressure")
            fileup.write(uf, step)
            fileup.write(pf, step)

end = time.time()
if MPI.rank(MPI.comm_world) == 0:
    print(
        "End of time loop. "
        f"Total computing time: {end - start}s, "
        f"average time per step: {(end - start)/N_STEPS:.3f}s."
    )

# End run postprocessing
postprocess(results)
