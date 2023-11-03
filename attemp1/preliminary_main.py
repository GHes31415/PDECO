import numpy as np
import dolfin as dl 
import matplotlib.pyplot as plt
import pickle 
import torch
import tqdm
import argparse
from solvers import *


# At some point, it'd be interesting to randomly sample Nx and Nt so that 
# every training example has a different number of sensors and evaluation points. 

def data_generation(num_test, Nx, Nt, gamma, delta, x0, x1, t0, t1, u0, filename):
    data = []
    for _ in tqdm.tqdm(range(num_test)):
        sensors_vals_1, sensors_vals_2, measurements = sol_diff_sys(x0, x1, t0, t1, u0, Nx, Nt, gamma, delta)
        data.append((sensors_vals_1,sensors_vals_2, measurements))
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
    file_training = f"data/heat_train_{num_train}.pkl"
    
    # Generate data set
    print('Generating data')
    data_generation(num_train, Nx, Nt, gamma, delta, x0, x1, t0, t1, u0, file_training)
    print('Training data generated')
    file_testing = f"data/heat_test_{num_test}.pkl"
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--Nx', type=int, default=100,help='Number of sensors in space')
    parser.add_argument('--Nt', type=int, default=100,help='Number of sensors in time')
    parser.add_argument('--gamma', type=float, default=0.1,help='Prior constant for generating GRF')
    parser.add_argument('--delta', type=float, default=0.5,help='Prior constant for generating GRF')
    parser.add_argument('--T', type=float, default=1,help='Final time')
    parser.add_argument('--num_train', type=int, default=80,help='Number of training examples')
    parser.add_argument('--num_test', type=int, default=20,help='Number of testing examples')

    args = parser.parse_args()
    Nx = args.Nx
    Nt = args.Nt
    gamma = args.gamma
    delta = args.delta
    T = args.T
    num_train = args.num_train
    num_test = args.num_test
    
    run( Nx,T, Nt, num_train, num_test,gamma,delta )


if __name__ == "__main__":
    main()