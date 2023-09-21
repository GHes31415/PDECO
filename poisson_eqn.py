'''Poisson equation solver'''

from dolfin import *


# Create mesh and defin function space

mesh = UnitSquareMesh(32,32)
#(, Family of polynomials, degree)
V = FunctionSpace(mesh,"Lagrange",1)

# Define boundary (x = 0 or x =1) returns boolean 

def boundary(x):
    return x[0] < DOLFIN_EPS or x[0]> 1.0 - DOLFIN_EPS

# Use the boundary to setup the DirichletBC 
u0 = Constant(0.0)
bc = DirichletBC(V,u0,boundary)

# Setup variational problem. 
# Trial function: u
# Test function: v
# To declare the input function f and boundary value g, we use Expression class.
# Declaration of bilinear form a and linear form L using UFL operatos

u = TrialFunction(V)
v = TestFunction(V)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)",degree = 2)
g = Expression("sin(5*x[0])",degree = 2)
a = inner(grad(u),grad(v))*dx
L = f*v*dx + g*v*ds

# Define a Function u to be teh solution .

u = Function(V)

solve(a== L,u,bc)

# u is modified on the call 

# Save solution in VTK format

file = File('poisson.pvd')
file << u

# Plot solution 
import matplotlib.pyplot as plt
plot(u)
plt.show()
