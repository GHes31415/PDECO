import numpy as np
import dolfin as dl 
import matplotlib.pyplot as plt
from solvers import *


solution = sol_diff_sys()

print(np.shape(solution))