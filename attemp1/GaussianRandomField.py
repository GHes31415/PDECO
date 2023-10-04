# create a folder HIPPYLIB/applications/GRF/ and add this file to it before running the code below

# Gaussian random field (GRF) model with covariance operator of the form
# (\delta I - \gamma \Delta)^{-\alpha}

import dolfin as dl
import ufl
import math
import numpy as np
import matplotlib.pyplot as plt
import argparse

import sys
import os
sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../../") )
from hippylib import *

def draw_sample(GRF):
    # initialize a vector of random noise
    noise = dl.Vector()
    GRF.init_vector(noise,"noise")
    # draw a sample from standard normal distribution
    parRandom.normal(1., noise)
    # initialize a sample of Gaussian random field (GRF)
    sample = dl.Vector()
    GRF.init_vector(sample, 0)
    # draw a sample from GRF distribution
    GRF.sample(noise, sample)
    return sample

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gaussian random field')
    parser.add_argument('--nd',
                        default=2,
                        type=int,
                        help="dimension d of physical domain D=(0,1)^d")
    parser.add_argument('--nx',
                        default=64,
                        type=int,
                        help="Number of elements in x-direction")
    parser.add_argument('--ny',
                        default=64,
                        type=int,
                        help="Number of elements in y-direction")
    parser.add_argument('--nz',
                        default=64,
                        type=int,
                        help="Number of elements in z-direction")
    parser.add_argument('--alpha',
                        default=2,
                        type=int,
                        help="(\delta I - \gamma \Delta)^{-\alpha}")
    parser.add_argument('--delta',
                        default=1,
                        type=float,
                        help="(\delta I - \gamma \Delta)^{-\alpha}")
    parser.add_argument('--gamma',
                        default=1,
                        type=float,
                        help="(\delta I - \gamma \Delta)^{-\alpha}")
    parser.add_argument('--nsamples',
                        default=3,
                        type=int,
                        help="Number of samples")
    parser.add_argument('--r',
                        default=64,
                        type=int,
                        help="Number of Karhunen-Loeve modes")
    args = parser.parse_args()

    try:
        dl.set_log_active(False)
    except:
        pass

    nd = args.nd
    nx = args.nx
    ny = args.ny
    nz = args.nz 
    if nd == 1:
        mesh = dl.UnitIntervalMesh(nx)
    elif nd == 2:
        mesh = dl.UnitSquareMesh(nx, ny)
    elif nd == 3:
        mesh = dl.UnitCubeMesh(nx, ny, nz)
    else:
        raise ValueError("Dimension d must be 1, 2, or 3")

    Vh = dl.FunctionSpace(mesh, 'Lagrange', 1)

    delta = args.delta
    gamma = args.gamma
    
    if args.alpha == 2:
        GRF = BiLaplacianPrior(Vh, gamma, delta, robin_bc=False)
    elif args.alpha == 1:
        GRF = LaplacianPrior(Vh, gamma, delta)

    # draw samples from GRF distribution
    for i in range(args.nsamples):

        filename = "figure/"+"d"+str(nd)+"alpha"+str(args.alpha)+"delta"+str(args.delta)+"gamma"+str(args.gamma)+"sample"+str(i)+".xdmf"
        with dl.XDMFFile(mesh.mpi_comm(), filename) as fid:
            fid.parameters["functions_share_mesh"] = True
            fid.parameters["rewrite_function_mesh"] = False 
            sample = draw_sample(GRF)
            fid.write(vector2Function(sample, Vh), i)

    # truncated Karhunen-Loeve expansion by double pass randomized algorithm
    r = args.r
    sample = dl.Vector()
    GRF.init_vector(sample, 0)
    Omega = MultiVector(sample, r+10)
    parRandom.normal(1., Omega)
    d, U = doublePass(Solver2Operator(GRF.Rsolver),
                        Omega, r, s = 1, check = False)

    # draw samples from truncated Karhunen-Loeve expansion with different number of truncation terms
    sample = dl.Vector()
    GRF.init_vector(sample, 0)
    y = np.random.normal(0, 1, U.nvec())
    filename = "figure/"+"d"+str(nd)+"alpha"+str(args.alpha)+"delta"+str(args.delta)+"gamma"+str(args.gamma)+"r"+str(args.r)+"KLsample"+".xdmf"
    KLsample = MultiVector(sample, r)
    with dl.XDMFFile(mesh.mpi_comm(), filename) as fid:
        fid.parameters["functions_share_mesh"] = True
        fid.parameters["rewrite_function_mesh"] = False 
        for i in range(U.nvec()):
            scale = np.sqrt(d[i])*y[i]
            sample.axpy(scale, U[i])        
            KLsample[i].set_local(sample)

    KLsample.export(Vh, filename, varname = "KLsample", normalize = True)

    # save eigenvalues and eigenvectors
    filename = "figure/"+"d"+str(nd)+"alpha"+str(args.alpha)+"delta"+str(args.delta)+"gamma"+str(args.gamma)+"r"+str(args.r)+"eigenvalues"+".dat"
    np.savetxt(filename,  d)
    filename = "figure/"+"d"+str(nd)+"alpha"+str(args.alpha)+"delta"+str(args.delta)+"gamma"+str(args.gamma)+"r"+str(args.r)+"eigenvectors"+".xdmf"
    U.export(Vh, filename, varname = "gen_evects", normalize = True)

    print("eigenvalues = ", d)
    plt.semilogy(d, '.-')
    filename = "figure/"+"d"+str(nd)+"alpha"+str(args.alpha)+"delta"+str(args.delta)+"gamma"+str(args.gamma)+"r"+str(args.r)+"eigenvalues"+".png"
    plt.ylabel(r"$\lambda_i$",fontsize=16)
    plt.xlabel(r"$i$",fontsize=16)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    # for i in np.arange(U.nvec()):
    #     pw_var.axpy(d[i], U[i]*U[i])
