'''
Dolfin implementation for solving heeat equation with Dirichled conditions. 
For details of the problem see sec. 3.1 Fenics the tutorial 

u' = Laplace(u) +f                  in the unit square
u = u_D                             on the boundary 
u = u_0                             at t = 0

u = 1 + x^2 + alpha*y^2 +beta*t
f = beta -2 -2*alpha
'''

import dolfin as dl 
import matplotlib.pyplot as plt

# Constants of the problem
alpha = 3; beta = 1.2
T = 2.0
num_steps = 10
dt = T/num_steps


#####Geometry of the problem

#Mesh
nx= ny = 8
mesh = dl.UnitSquareMesh(nx,ny)

#Functional space
V = dl.FunctionSpace(mesh,'P',1)


#Boundary
u_D = dl.Expression('1+x[0]*x[0]+alpha*x[1]*x[1]+beta*t',degree = 2, alpha = alpha, beta = beta, t = 0)

def boundary(x,on_boundary):
    return on_boundary

bc = dl.DirichletBC(V,u_D,boundary)

# Define initial value

u_n = dl.project(u_D,V)

# Define the variational problem 

u = dl.TrialFunction(V)
v = dl.TrialFunction(V)
f = dl.Constant(beta-2-2*alpha)

F = u*v*dl.dx + dt*dl.dot(dl.grad(u),dl.grad(v))*dl.dx -(u_n+dt*f)*v*dl.dx
a,L = dl.Lhs(F),dl.Rhs(F)
