import numpy as np
import dolfin as dl 
import matplotlib.pyplot as plt
import pickle 
import torch
import tqdm
from solvers import *


# At some point, it'd be interesting to randomly sample Nx and Nt so that 
# every training example has a different number of sensors and evaluation points. 

def data_generation(num_test, Nx, Nt, gamma, delta, x0, x1, t0, t1, u0, filename):
    data = []
    for i in tqdm.tqdm(range(num_test)):
        sensors_vals_1, sensors_vals_2, measurements = sol_diff_sys(x0, x1, t0, t1, u0, Nx, Nt, gamma, delta)
        data.append((torch.tensor(sensors_vals_1), torch.tensor(sensors_vals_2), torch.tensor(measurements)))
    torch.save(data, filename)

def run( Nx,T, Nt, num_train, num_test,gamma= 0.1,delta = 0.5):
    #Sensors in space Nx
    #Sensors in time Nt

    #This command is for avoinding to many prints in terminal
    dl.set_log_level(50)

    #Spatial and time domain
    x0,x1 = 0,1
    t0,t1 = 0,T
    #Initial condition
    u0 = dl.Constant(0.0)#dl.Expression('sin(2*pi*x[0])',degree = 2)
    #Data for trunk network
    # xt = [(x, y) for x in np.linspace(x0, x1, Nx+1) for y in np.linspace(0, T, Nt+1)]
    #Data for training and testing
    file_training = "data/DR_train.pkl"
    
    # Generate data set
    print('Generating data')
    data_generation(num_train, Nx, Nt, gamma, delta, x0, x1, t0, t1, u0, file_training)
    print('Training data generated')
    file_testing = "data/DR_test.pkl"
    # Generate testing data set
    data_generation(num_test, Nx, Nt, gamma, delta, x0, x1, t0, t1, u0, file_testing)
    print('Testing data generated')

    # data_train = {}
    # data_generation(num_train, Nx, Nt, gamma, delta, x0, x1, t0, t1, u0,data_train)
    # data_train['xt'] = xt
    # pickle.dump(data_train, open("data/DR_train.pkl", "wb"))
    # data_test = {}
    # data_generation(num_test, Nx, Nt, gamma, delta, x0, x1, t0, t1, u0,data_test)
    # data_test['xt'] = xt
    # data_test['xt'] = xt
    # pickle.dump(data_test, open("data/DR_test.pkl", "wb"))
    
    


def main():

    Nx = 50
    gamma = 0.1
    delta = 0.5
    T = 1
    Nt = 25
    num_train = 80
    num_test = 20

    run( Nx,T, Nt, num_train, num_test,gamma,delta )


if __name__ == "__main__":
    main()