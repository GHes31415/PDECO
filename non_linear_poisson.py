'''
Solution of non linear poisson equation
'''
import matplotlib.pyplot as plt
from dolfin import *

# Boundary conditions



# Mesh and Function space

mesh = UnitSquareMesh(32,32)
File('mesh.pvd')<<mesh

# CG == Lagrange (function space)

V = FunctionSpace(mesh,"CG",1)

# Subdomain for Dirichlet boundary condition 

class DirichletBoundary(SubDomain):
    def inside(self,x,on_boundary):
        return abs(x[0]-1.0)< DOLFIN_EPS and on_boundary
    
g = Constant(1.0)

# Remeber, input (fn space, fun, Bool for boundary )
bc = DirichletBC(V,g,DirichletBoundary())

# Definition of variational problem 

u = Function(V)
v = TestFunction(V)
f = Expression("kappa*x[0]*sin(x[1])",kappa = 1,degree = 2)
F = inner((1+u**2)*grad(u),grad(v))*dx -f*v*dx

# Compute solution 

solve(F == 0, u,bc,
        solver_parameters={"newton_solver":{"relative_tolerance":1e-6}})

# Plot solution

plt.figure()
plot(u,title= "Solution")

plt.figure()
plot(grad(u),title= 'Solution gradient')

plt.show()

file = File('nonlinear_poisson.pvd')
file << u


