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
import numpy as np 

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
u_D = dl.Expression('1+x[0]*x[0]+alpha*x[1]*x[1]+beta*t',
                    degree = 2, alpha = alpha, beta = beta, t = 0)

def boundary(x,on_boundary):
    return on_boundary

bc = dl.DirichletBC(V,u_D,boundary)

# Define initial value

u_n = dl.interpolate(u_D,V)

# Define the variational problem 

u = dl.TrialFunction(V)
v = dl.TestFunction(V)
'''
By mistake I typed TrialFunction for v, instead of TestFunction and I got the following error
raise NotImplementedError("Cannot take length of non-vector expression.")
NotImplementedError: Cannot take length of non-vector expression.
'''
f = dl.Constant(beta-2-2*alpha)

F = u*v*dl.dx + dt*dl.dot(dl.grad(u),dl.grad(v))*dl.dx -(u_n+dt*f)*v*dl.dx
# Dolfin figures out which terms belon to the lhs and rhs
a,L = dl.lhs(F),dl.rhs(F)
#a = u*v*dl.dx + dt*dl.dot(dl.grad(u),dl.grad(v))*dl.dx
L = (u_n+dt*f)*v*dl.dx
u = dl.Function(V)



# Time-stepping
t = 0 
for n in range(num_steps):

    #Update current time
    t+= dt
    u_D.t = t

    #Compute solution 
    dl.solve(a==L,u,bc)
    #Interestingly in this problem we cannot do dl.solve(F==0,u,bc) dunno why 

    dl.plot(u)
    plt.show()

    # Compute error at vertices 

    u_e = dl.interpolate(u_D,V)
    error = np.abs(u_e.vector()-u.vector()).max()
    print('t = %.2f:error = %.3g'%(t,error))

    #Update previous solution
    u_n.assign(u)

# Hold plot 
# dl.interactive()
plt.show()


