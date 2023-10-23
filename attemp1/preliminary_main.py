import numpy as np
import dolfin as dl 
import matplotlib.pyplot as plt
import pickle 
from solvers import *


def data_generation(num_test, Nx, Nt, gamma, delta, x0, x1, t0, t1, u0,dict):
    for i in range(num_test):

        sensors_vals_1,sensors_vals_2,measurements = sol_diff_sys(x0,x1,t0,t1,u0,Nx,Nt,gamma, delta)
        dict[i] = {'sensors': [sensors_vals_1,sensors_vals_2], 'solution':measurements}



def run( Nx,T, Nt, num_train, num_test,gamma= 0.1,delta = 0.5):
    #Sensors in space Nx
    #Sensors in time Nt

    #This command is for avoinding to many prints in terminal
    dl.set_log_level(30)

    #Spatial and time domain
    x0,x1 = 0,1
    t0,t1 = 0,T
    #Initial condition
    u0 = dl.Constant(0.0)#dl.Expression('sin(2*pi*x[0])',degree = 2)
    #Data for trunk network
    xt = [(x, y) for x in np.linspace(x0, x1, Nx+1) for y in np.linspace(0, T, Nt+1)]
    #Data for training and testing
    data_train = {}
    data_generation(num_train, Nx, Nt, gamma, delta, x0, x1, t0, t1, u0,data_train)
    data_train['xt'] = xt
    pickle.dump(data_train, open("data/DR_train.pkl", "wb"))
    data_test = {}
    data_generation(num_test, Nx, Nt, gamma, delta, x0, x1, t0, t1, u0,data_test)
    data_test['xt'] = xt
    data_test['xt'] = xt
    pickle.dump(data_test, open("data/DR_test.pkl", "wb"))
    
    


def main():

    Nx = 10
    gamma = 0.1
    delta = 0.5
    T = 1
    Nt = 10
    num_train = 800
    num_test = 200

    run( Nx,T, Nt, num_train, num_test,gamma,delta )


if __name__ == "__main__":
    main()