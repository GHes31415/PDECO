'''
Dolfin implementation for training data. 
'''


import dolfin as dl 
import matplotlib.pyplot as plt
import numpy as np 
import hippylib as hp
import ufl 
import logging
# import pdb
# pdb.set_trace()

logging.getLogger('FFC').setLevel(logging.WARNING)
dl.set_log_level(dl.LogLevel.WARNING)
'''
Problem 1

Diffusion system with reaction term 

u_t = (D(x)u_x)_x + g(t)                [0,1]x[0,1]
u   = 0                                 [0,1]x[0]
u   = 0                                 [0]x[0,1]u[1]x[0,1] 

D(x) = 0.01(|f(x)|+1)

f(x) and g(t) are GRF with laplacian prior 

The domain is partition in a mesh of 100x100

'''


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


def sol_diff_sys(x0,x1,t0,t1,u0,nx,nt,gamma = 0.1, delta = 0.5):
    '''
    x0,x1       = spatial domain 
    t0,t1       = time domain
    u0          = dolfin function, initial condition 
    nx,nt       = spatial and time partition
    gamma,delta = prior constants for generating GRF
    '''
    
    # t = np.linspace(t0, t1, nt)
    mesh_space = dl.IntervalMesh(nx,x0,x1)
    mesh_time = dl.IntervalMesh(nt,t0,t1)

    dt = (t1-t0)/nt

    #Functional space
    V = dl.FunctionSpace(mesh_space,'CG',1)
    V_t = dl.FunctionSpace(mesh_time,'P',1)

    #Random functions
    GRF = hp.LaplacianPrior(V, gamma, delta)
    GRF_t = hp.LaplacianPrior(V_t,gamma,delta)
    sample_d = draw_sample(GRF)
    sample_g = draw_sample(GRF_t)
    # at the time being I'm going to save the samples directly. 
    # The exponential funciton is biyective and there hopefully
    # won't be loss of generality by saving the samples
    sensor_1 = sample_d.get_local()
    sensor_2 = sample_g.get_local()
    # Now d and g can be evaluated as functions
    d = hp.vector2Function(sample_d,V)
    exp_D= ufl.exp(d)
    g = hp.vector2Function(sample_g,V_t)

    #Boundary 

    
    # def boundary(x,on_boundary):
    #     return on_boundary
    
    # u_D = u0
    bc  = dl.DirichletBC(V,u0,'on_boundary')

    # Initial value

    u_n = dl.interpolate(u0,V)
    
    # Define the variational problem 
    u = dl.TrialFunction(V)
    v = dl.TestFunction(V)
    t = t0
    # Weak formulation do ti explicitly
    space_g = dl.Function(V)
    a = u*v*dl.dx+dt*(exp_D)*dl.inner(dl.grad(u),dl.grad(v))*dl.dx 
    L = (u_n + dt*space_g)*v*dl.dx
    
    A,b = dl.assemble_system(a,L,bc)    
    # mumps: multifrontal massively parallel sparse direct solver
    solver = dl.LUSolver(A,'mumps')
    
    

    u = dl.Function(V)
    # Solve solution of PDE
    final_u = np.zeros((nx+1,nt+1))
    
    final_u[:,0] = u_n.vector().get_local()
    

    for n in range(1,nt+1):

        #Update current time
        t = t0 + n*dt

        # if time_dep_boundary:
        #     u_D.t = t
        # update the value of the rhs , value g(t_{n+1})
        space_g.assign(dl.Constant(g(t)))
        b = dl.assemble(L)
    
        #Compute solution 
        solver.solve(A,u.vector(),b)

        #save
        final_u[:,n] = u.vector().get_local()
        
        #Update previous solution
        u_n.assign(u)
    
    # Use fenics data structure to save data, check ad_diff in hippylib specially for large scale probls
    return sensor_1,sensor_2,final_u.reshape(((nx+1)*(nt+1),))

def sol_diff_plot(gamma = 0.1, delta = 0.5, experiment = 0,plot = False,time_dep_boundary = lambda t:0):
    '''
    
    '''
    
    # Constants of the problem 
    T = 1.0
    num_steps = 10
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
    if plot:
        plt.plot(sample_d.get_local())
        plt.savefig(f'plots/d_exp_{experiment}.png')
        plt.show()
        plt.close()
        plt.plot(sample_g.get_local())
        plt.savefig(f'plots/g_exp_{experiment}.png')
        plt.show()
        plt.close()
    d = hp.vector2Function(sample_d,V)
    exp_D= ufl.exp(d)
    g = hp.vector2Function(sample_g,V)

    
    

    #Boundary 

    
    def boundary(x,on_boundary):
        return on_boundary
    u_D = dl.Expression('sin(2*pi*x[0])',degree = 2)#dl.Constant(0.0)
    bc  = dl.DirichletBC(V,u_D,boundary)

    # Initial value

    u_n = dl.interpolate(u_D,V)

    '''Poisson solver for verifying steady state'''

    # u = dl.TrialFunction(V)
    # v = dl.TestFunction(V)
    # # Solving poisson 
    # a = (exp_D)*dl.inner(dl.grad(u),dl.grad(v))*dl.dx
    # L = g*v*dl.dx

    # u = dl.Function(V)

    # dl.solve(a== L,u,bc)

    # dl.plot(u)
    # plt.savefig(f'plots/sol_poiss_exp{experiment}.png')
    # plt.close()


    # Define the variational problem 
    u = dl.TrialFunction(V)
    v = dl.TestFunction(V)

    # Weak formulation
    a = u*v*dl.dx+dt*(exp_D)*dl.inner(dl.grad(u),dl.grad(v))*dl.dx
    L = (u_n + dt*g)*v*dl.dx  

    u = dl.Function(V)

    final_u = np.zeros((nx+1,num_steps))
    
    final_u[:,0] = u_n.vector().get_local()
    t = 0

    for n in range(num_steps):

        #Update current time
        t+= dt

        if time_dep_boundary:
            u_D.t = t

        #Compute solution 
        dl.solve(a==L,u,bc)

        #save

        final_u[:,n] = u.vector().get_local()

        

        
        if n%10 ==0:
            dl.plot(u_n)
            plt.show()

        #Update previous solution
        u_n.assign(u)


    solution = final_u
    
    # dl.plot(u)
    # plt.savefig(f'plots/sol_exp_{experiment}.png')
    # plt.show()
    # plt.close()
    # solution = time_evolution(num_steps,dt, a,L,u,bc,u_D,V,time_dep_boundary= False,plot = True)

    return solution





    
