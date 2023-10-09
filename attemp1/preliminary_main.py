import numpy as np
import dolfin as dl 
import matplotlib.pyplot as plt
from solvers import *

def run( Nx,T, Nt, num_train, num_test,gamma= 0.1,delta = 0.5):
    #Sensors in space Nx
    #Sensors in time Nt

    #This command is for avoinding to many prints in terminal
    dl.set_log_level(30)

    x0,x1 = 0,1
    t0,t1 = 0,T
    

    u0 = dl.Constant(0.0)#dl.Expression('sin(2*pi*x[0])',degree = 2)


    measurements = np.zeros((num_train,(Nx+1)*Nt))
    sensors_vals_1 = np.zeros((num_train,Nx+1))
    sensors_vals_2 = np.zeros((num_train,Nt+1))

    

    for i in range(num_train):

        sensors_vals_1[i,:],sensors_vals_2[i,:],measurements[i,:] = sol_diff_sys(x0,x1,t0,t1,u0,Nx,Nt,gamma, delta)
    #this will have to change eventually. The xt point will have to be given from the code
    #i'll eventually sample them for each iteration
    xt = [(x, y) for x in np.linspace(x0, x1, Nx) for y in np.linspace(0, T, Nt)]

    X_train, y_train = (sensors_vals_1, sensors_vals_2, xt), measurements

    measurements = np.zeros((num_test,(Nx+1)*Nt))
    sensors_vals_1 = np.zeros((num_test,Nx+1))
    sensors_vals_2 = np.zeros((num_test,Nx+1))

    for i in range(num_test):

        sensors_vals_1[i,:],sensors_vals_2[i,:],measurements[i,:]  = sol_diff_sys(x0,x1,t0,t1,u0,Nx,Nt,gamma, delta)

    

    X_test, y_test = (sensors_vals_1, sensors_vals_2, xt), measurements

    print(type(X_train))

    # np.savez_compressed("data/DR_train.npz", X_train=X_train, y_train=y_train)
    # np.savez_compressed("data/DR_test.npz", X_test=X_test, y_test=y_test)



def main():

    Nx = 100
    gamma = 0.1
    delta = 0.5
    T = 1
    Nt = 100
    num_train = 1
    num_test = 0

    run( Nx,T, Nt, num_train, num_test,gamma,delta )


if __name__ == "__main__":
    main()