'''
Dolfin implementation for training data. 
'''


import dolfin as dl 
import matplotlib.pyplot as plt
import numpy as np 
import hippylib as hp
import ufl 

'''
Problem 1

Diffusion system with reaction term 

u_t = (D(x)u_x)_x + g(x)                [0,1]x[0,1]
u   = 0                                 [0,1]x[0]
u   = 0                                 [0]x[0,1]u[1]x[0,1] 

D(x) = 0.01(|f(x)|+1)

f(x) and g(x) are GRF with laplacian prior 

The domain is partition in a mesh of 100x100

'''

def time_evolution(num_steps,dt, a,L,u,bc,u_D,V,time_dep_boundary= False,plot = False):
    # Time-stepping
    u_n = dl.interpolate(u_D,V)

    u0 = u_n.vector().get_local()

    nx = len(u0)
    
    t = 0 

    final_u = np.zeros((nx,num_steps))
    print(np.shape(final_u))
    final_u[:,0] = u0

    for n in range(num_steps):

        #Update current time
        t+= dt
        if time_dep_boundary:
            u_D.t = t

        #Compute solution 
        dl.solve(a==L,u,bc)

        final_u[:,n] = u.vector().get_local()

        #save
        if plot:
            dl.plot(u)
            plt.show()

        #Update previous solution
        u_n.assign(u)

    return final_u


def draw_sample(GRF):
    # initialize a vector of random noise
    noise = dl.Vector()
    GRF.init_vector(noise,"noise")
    # draw a sample from standard normal distribution
    hp.parRandom.normal(1., noise)
    # initialize a sample of Gaussian random field (GRF)
    sample = dl.Vector()
    GRF.init_vector(sample, 0)
    # draw a sample from GRF distribution
    GRF.sample(noise, sample)
    return sample


def sol_diff_sys(gamma = 0.1, delta = 0.5):
    '''
    
    '''
    
    # Constants of the problem 
    T = 1.0
    num_steps = 100
    dt = T/num_steps

    # Geometry of the problem 

    #Mesh 
    nx = 100 
    mesh = dl.UnitIntervalMesh(nx)
    #

    #Functional space
    V = dl.FunctionSpace(mesh,'P',1)

    #Random functions
    GRF = hp.LaplacianPrior(V, gamma, delta)
    sample_d = draw_sample(GRF)
    sample_g = draw_sample(GRF)

    d = hp.vector2Function(sample_d,V)
    exp_D= 0.1*ufl.exp(d)
    g = hp.vector2Function(sample_g,V)

    #Boundary 

    
    def boundary(x,on_boundary):
        return on_boundary
    u_D = dl.Constant(0.0)
    bc  = dl.DirichletBC(V,u_D,boundary)

    # Initial value

    u_n = dl.interpolate(u_D,V)

    # Define the variational problem 

    u = dl.TrialFunction(V)
    v = dl.TestFunction(V)

    # Weak formulation
    a = (exp_D)*dl.inner(dl.grad(u),dl.grad(v))*dl.dx
    L = g*v*dl.dx  

    u = dl.Function(V)

    solution = time_evolution(num_steps,dt, a,L,u,bc,u_D,V,time_dep_boundary= False,plot = True)

    return solution





    
