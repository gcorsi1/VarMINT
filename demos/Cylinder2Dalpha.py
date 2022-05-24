"""
This demo solves the 2D flow past a cylinder, and calculates lift and drag 
coefficients that can be compared with literature, see 
https://dx.doi.org/10.1016/j.cma.2014.10.040
"""
from VarMINT import *
from VarMINTpostproc import calc_force_coeffs_turek
import math
import numpy as np
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

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
print(f"Mesh has hmax = {mesh.hmax()}, hmin = {mesh.hmin()}.")

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


class BoundaryFunction(UserExpression):
    def __init__(self, t, ramp_end_t, **kwargs):
        super().__init__(**kwargs)
        self.t = t
        self.ramp_end_t = ramp_end_t

    # simple linear time ramp
    def eval(self, values, x):
        U = 1.0 * (self.t - self.ramp_end_t) if self.t < self.ramp_end_t else 1.0
        V = 0.01 if self.t < self.ramp_end_t else 0.0  # small disturbance 
        values[0] = U
        values[1] = V

    def value_shape(self):
        return (2,)


# Weak problem residual; note use of alpha intermediate velocity
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
)

# Project the initial condition:
u_IC = as_vector((Constant(0.0), Constant(0.0)))
up_old.assign(project(as_vector((u_IC[0], u_IC[1], Constant(0.0))), V))

# Init all to zero
for ui in (up, upt, upt_old):
    ui.assign(up_old)


# Set no-penetration BCs on velocity and pin down pressure in one corner:
obstacle = 1
inflow = 2
outflow = 3
bottom = 4
top = 5

# Define boundary conditions
U_inlet = BoundaryFunction(0.0, ramp_end_t=0.5)
bc_inlet = DirichletBC(V.sub(0), U_inlet, boundaries, inflow)
bc_bottom = DirichletBC(V.sub(0).sub(1), Constant(0.0), boundaries, bottom)
bc_top = DirichletBC(V.sub(0).sub(1), Constant(0.0), boundaries, top)
bc_obstacle = DirichletBC(V.sub(0), Constant((0.0, 0.0)), boundaries, obstacle)
bc_outlet = DirichletBC(V.sub(1), Constant(0.0), boundaries, outflow)
bcs = [bc_inlet, bc_bottom, bc_top, bc_obstacle, bc_outlet]

# define the functions that will be used to integrate the forces on the
# cylinder variationally, these will be equal to a cartesian base vector on
# the surface of the cylinder, and 0 on the other Dirichlet boundaries
VM = V.sub(0).collapse()
QM = V.sub(1).collapse()
bc_m_inlet = DirichletBC(VM, Constant((0.0, 0.0)), boundaries, inflow)
bc_m_bottom = DirichletBC(VM, Constant((0.0, 0.0)), boundaries, bottom)
bc_m_top = DirichletBC(VM, Constant((0.0, 0.0)), boundaries, top)
bc_m_obstacle_e0 = DirichletBC(VM, Constant((1.0, 0.0)), boundaries, obstacle)
bc_m_obstacle_e1 = DirichletBC(VM, Constant((0.0, 1.0)), boundaries, obstacle)
bcs_m = [bc_m_inlet, bc_m_bottom, bc_m_top]
z = TrialFunction(VM)
s = TestFunction(VM)
a_m = inner(grad(z), grad(s)) * dx
f_m = dot(Constant((0.0, 0.0)), s) * dx
eis = [Function(VM) for _ in range(mesh.geometry().dim())]

# solve an auxiliary laplacian problem in order to calculate the e_i
# and define a zero function to be used for the pressure test function
for eii, bci in zip(eis, [bc_m_obstacle_e0, bc_m_obstacle_e1]):
    solve(a_m == f_m, eii, bcs=bcs_m + [bci])
q0 = Function(QM)
q0.assign(project(Constant(0.0), QM))

t = 0.0
results["t"].append(t)

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
)

results["CD"].append(2.0 * assemble(F0))
results["CL"].append(2.0 * assemble(F1))
results["freq"].append(0.0)

# Time stepping loop:
with XDMFFile("solu.xdmf") as fileu, XDMFFile("solp.xdmf") as filep:
    for step in range(0, N_STEPS):
        t += float(Dt)
        print("======= Time step " + str(step + 1) + "/" + str(N_STEPS) + " =======")

        # Predictor step, velocity assumed to be the same, notice that it
        # is ut_old that is used in the calculation of the intermediate alpha
        # variables
        # upt_old.assign((gamma_f - 1.0) / gamma_f * upt)
        upt_old.assign(upt)

        # Update dirichlet boundary condition:
        U_inlet.t = t
        solve(F == 0, up, bcs=bcs)

        # Corrector step
        upt.assign(
            1.0 / float(Dt) / gamma_f * (up - up_old)
            - (1.0 - gamma_f) / gamma_f * upt_old
        )
        up_old.assign(up)

        uf, pf = up.split(deepcopy=True)
        uf.rename("Velocity", "Velocity")
        pf.rename("Pressure", "Pressure")
        if not step % 20:
            fileu.write(uf, step)
            filep.write(pf, step)

        results["t"].append(t)
        results["CD"].append(2.0 * assemble(F0))
        results["CL"].append(2.0 * assemble(F1))

        # calculate the main frequency of oscillation
        Ntot = len(results["CL"])
        ws = np.fft.fft(results["CL"])
        Ttot = results["t"][-1]
        dfs = 1.0 / Ttot
        freqs = np.fft.fftfreq(Ntot) * Ntot * dfs
        idx = np.argmax(np.abs(ws))
        maxf = np.abs(freqs[idx])
        if (MPI.rank(MPI.comm_world) == 0):
            print(f"main frequency sampled from data is: {maxf:.2f}")
        results["freq"].append(maxf)

print("End of time loop.")

# end run postprocessing

data_preproc = pd.DataFrame.from_dict(results)
data_preproc.to_csv("./raw_data.csv")

fade_ = 1  # DOWNSAMPLE FACTOR FOR THE PLOTS
data_preproc = data_preproc.iloc[1::fade_, :]  # only get some of the rows in the plot
# for plots, remove initial seconds where pressure wave gives spurious 
# very high values
data_preproc = data_preproc[data_preproc.t > 3]
g = sns.lineplot( x="t", y="value", hue="variable", hue_order=["CL", "CD"], marker="o", data=pd.melt(data_preproc, ["t"]),)
# g = sns.lineplot(x="t", y="Fy", marker="o", data=data_preproc)
# g = sns.lineplot(x="t", y="Fx", marker="o", data=data_preproc)
sns.despine()
plt.xlabel(r"$t [s]$")
plt.ylabel("")
plt.legend(
    title="Coefficient",
    bbox_to_anchor=(0.9, 0.9),
    loc=2,
    labels=[r"$C_L$", r"$C_D$"],
)
plt.savefig("./coefficients.pdf")
plt.close()
