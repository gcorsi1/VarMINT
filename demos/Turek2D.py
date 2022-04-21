"""
This demo solves the 2D Taylor--Green vortex problem, illustrating usage
in unsteady problems and demonstrating spatio-temporal convergence under 
quasi-uniform space--time refinement.  
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
parser.add_argument('--Nel',dest='Nel',default=16,
                    help='Number of elements in each direction.')
parser.add_argument('--Re',dest='Re',default=100.0,
                    help='Reynolds number.')
parser.add_argument('--k',dest='k',default=1,
                    help='Polynomial degree.')
parser.add_argument('--T',dest='T',default=1.0,
                    help='Length of time interval to consider.')

args = parser.parse_args()
Nel = int(args.Nel)
Re = Constant(float(args.Re))
k = int(args.k)
T = float(args.T)

# store results
results = defaultdict(list)
####### Analysis #######

# Avoid overkill automatically-determined quadrature degree:
QUAD_DEG = 2*k
dx = dx(metadata={"quadrature_degree":QUAD_DEG})

# Domain:
mesh = Mesh(MPI.comm_world)
with XDMFFile("mesh_turekCFD.xdmf") as file:
    file.read(mesh)

mtot_ = MPI.sum(MPI.comm_world, mesh.num_cells())
print(f"Read mesh with {mtot_} cells.")
print(f"Mesh has hmax = {mesh.hmax()}, hmin = {mesh.hmin()}.")

# Read boundary data
mvc_boundaries = MeshValueCollection("size_t", mesh, mesh.topology().dim()-1)
with XDMFFile("mf_turekCFD.xdmf") as file:
    file.read(mvc_boundaries)
boundaries = cpp.mesh.MeshFunctionSizet(mesh, mvc_boundaries)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
n = FacetNormal(mesh)

# Mixed velocity--pressure space:
V = equalOrderSpace(mesh,k=k)

# Solution and test functions:
up = Function(V)
u,p = split(up)
vq = TestFunction(V)
v,q = split(vq)

# Midpoint time integration:
N_STEPS = Nel # Space--time quasi-uniformity
Dt = Constant(T/N_STEPS)
up_old = Function(V)
u_old,_ = split(up_old)
u_mid = 0.5*(u+u_old)
u_t = (u-u_old)/Dt

# Determine physical parameters from Reynolds number:
rho = Constant(1.0)
mu = Constant(1.0/Re)
nu = mu/rho

# Initial condition for the Taylor--Green vortex
x = SpatialCoordinate(mesh)
u_IC = as_vector((Constant(0.0),Constant(0.0)))

class BoundaryFunction(UserExpression):
    def __init__(self, t, **kwargs):
        super().__init__(**kwargs)
        self.t = t

    def eval(self, values, x):
        U = 1.5*sin(pi*self.t/8)
        values[0] = 4*U*x[1]*(0.41-x[1])/pow(0.41, 2)
        values[1] = 0

    def value_shape(self):
        return (2,)

# Weak problem residual; note use of midpoint velocity:
F = interiorResidual(u_mid,p,v,q,rho,mu,mesh,
                     u_t=u_t,Dt=Dt,C_I=Constant(6.0*(k**4)),dx=dx)

uext = Expression(("1.00","0.0"), degree=2)
n_analytic = Expression(("x[0]", "x[1]"), degree=3)

# F += weakDirichletBC(u,p,v,q,uext,
#                      rho,mu,mesh,
#                      ds=ds(2),
#                      n_analytic=n_analytic,
#                      sym=True,C_pen=Constant(1e3),
#                      overPenalize=False)

# Project the initial condition:
up_old.assign(project(as_vector((u_IC[0],u_IC[1],Constant(0.0))),V))

# Same-velocity predictor:
up.assign(up_old)

# Set no-penetration BCs on velocity and pin down pressure in one corner:
inflow = 2
outflow = 3
walls = 4
obstacle = 5

# Define boundary conditions
U_inlet = BoundaryFunction(0.0)
bc_inlet = DirichletBC(V.sub(0), U_inlet, boundaries, inflow)
bc_walls = DirichletBC(V.sub(0), Constant((0.0,0.0)), boundaries, walls)
bc_obstacle = DirichletBC(V.sub(0), Constant((0.0,0.0)), boundaries, obstacle)
bc_outlet = DirichletBC(V.sub(1), Constant(0.0), boundaries, outflow)
bcs = [bc_inlet, bc_walls, bc_obstacle, bc_outlet]
# corner_str = "near(x[0], -0.99) && near(x[1], 0.0)"
# bcs = [DirichletBC(V.sub(0), Constant((0.0, 0.0)), boundaries, 1),
#        DirichletBC(V.sub(1), Constant(0.0), corner_str,"pointwise")]

t = 0.0
results['ts'].append(t)
calc_force_coeffs(u,p,mu,n,ds(5),results,len_scale=0.1)

# Time stepping loop:
with XDMFFile("solu.xdmf") as fileu, XDMFFile("solp.xdmf") as filep:
    for step in range(0,N_STEPS):
        t += float(Dt)
        print("======= Time step "+str(step+1)+"/"+str(N_STEPS)+" =======")

        # Update dirichlet boundary condition:
        U_inlet.t = t
        solve(F==0,up,bcs=bcs)
        up_old.assign(up)

        uf, pf = up.split(deepcopy=True)
        uf.rename("Velocity","Velocity")
        pf.rename("Pressure","Pressure")
        if not step % 200:
            fileu.write(uf, step)
            filep.write(pf, step)

        results['ts'].append(t)
        calc_force_coeffs(u,p,mu,n,ds(5),results,len_scale=0.1)
    np.savez("resultsTUREK", CD=np.array(results['c_ds']), CL=np.array(results['c_ls']), t=np.array(results['ts']))

print("End of time loop.")
