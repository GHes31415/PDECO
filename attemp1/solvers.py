'''
Dolfin implementation for training data. 
'''


import dolfin as dl 
import matplotlib.pyplot as plt
import numpy as np 

'''
Problem 1

Diffusion system with reaction term 

u_t = (D(x)u_x)_x + g(x)                [0,1]x[0,1]
u   = 0                                 [0,1]x[0]
u   = 0                                 [0]x[0,1]u[1]x[0,1] 

D(x) = 0.01(|f(x)|+1)

f(x) and g(x) are GRF with l =0.2

The domain is partition in a mesh of 100x100

'''


def sol_diff_sys(f,g):
    '''
    f:              np array 100
    g:              np array 100
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

    #Boundary 

    u_D = dl.Constant(0.0)
    def boundary(x,on_boundary):
        return on_boundary
    bc  = dl.DirichletBC(V,u_D,boundary)

    # Initial value

    u_n = dl.interpolate(u_D,V)

    # Define the variational problem 

    u = dl.TrialFunction(V)
    v = dl.TestFunction(V)

    D = dl.Expression('0.001')
