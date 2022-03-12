"""
This demo solves the 2D Taylor--Green vortex problem, illustrating usage
in unsteady problems and demonstrating spatio-temporal convergence under 
quasi-uniform space--time refinement.  
"""
from VarMINT import *

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

####### Analysis #######

# Avoid overkill automatically-determined quadrature degree:
QUAD_DEG = 2*k
dx = dx(metadata={"quadrature_degree":QUAD_DEG})

# Domain:
import math
mesh = RectangleMesh(Point(-math.pi,-math.pi),
                     Point(math.pi,math.pi),
                     Nel,Nel)

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
u_IC = as_vector((sin(x[0])*cos(x[1]),-cos(x[0])*sin(x[1])))

# Time dependence of exact solution, evaluated at time T:
solnT = exp(-2.0*nu*T)
u_exact = solnT*u_IC

# Expression for boundary condition:
class solexp(UserExpression):
    def __init__(self, nu, **kwargs):
        super().__init__(kwargs)
        self.t = 0.0
        self.nu = nu
    def eval(self, values, x):
        texp = exp(-2.0*self.nu*self.t)
        values[0] = sin(x[0])*cos(x[1])*texp
        values[1] = -cos(x[0])*sin(x[1])*texp
    def value_shape(self):
        return (2,)
solt = solexp(nu, degree=2)

# Weak problem residual; note use of midpoint velocity:
F = interiorResidual(u_mid,p,v,q,rho,mu,mesh,
                     u_t=u_t,Dt=Dt,C_I=Constant(6.0*(k**4)),dx=dx)

F += weakDirichletBC(u,p,v,q,solt,
                     rho,mu,mesh,
                     ds=ds,
                     sym=True,C_pen=Constant(1e3),
                     overPenalize=False)

# Project the initial condition:
up_old.assign(project(as_vector((u_IC[0],u_IC[1],Constant(0.0))),V))

# Same-velocity predictor:
up.assign(up_old)

# Set no-penetration BCs on velocity and pin down pressure in one corner:
corner_str = "near(x[0],-pi) && near(x[1],-pi)"
bcs = [DirichletBC(V.sub(1),Constant(0.0),corner_str,"pointwise")]
# bcs = [DirichletBC(V.sub(0).sub(0),Constant(0.0),
#                    "near(x[0],-pi) || near(x[0],pi)"),
#        DirichletBC(V.sub(0).sub(1),Constant(0.0),
#                    "near(x[1],-pi) || near(x[1],pi)"),
#        DirichletBC(V.sub(1),Constant(0.0),corner_str,"pointwise")]

t = 0.0
# Time stepping loop:
for step in range(0,N_STEPS):
    t += Dt
    print("======= Time step "+str(step+1)+"/"+str(N_STEPS)+" =======")

    # Update dirichlet boundary condition:
    solt.t = t
    solve(F==0,up,bcs=bcs)
    up_old.assign(up)

uf, pf = up.split(deepcopy=True)
uf.rename("Velocity","Velocity")
pf.rename("Pressure","Pressure")
with XDMFFile("sol.xdmf") as file:
    file.write(uf, 0)
    file.write(pf, 1)

# Check error:
def L2Norm(u):
    return math.sqrt(assemble(inner(u,u)*dx))
e_u = u - u_exact
print("Element size = "+str(2.0*math.pi/Nel))
print("H1 seminorm velocity error = "+str(L2Norm(grad(e_u))))
print("L2 norm velocity error = "+str(L2Norm(e_u)))
