'''
This file is the main configuration module. 
'''

import numpy as np
from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.constraints import LinearInequalityConstraint

## Parameter space:

# Parameter space bounds:
space = ParameterSpace([
    ContinuousParameter('wmax', 0.1, 1.5),
    ContinuousParameter('wmin', 0.03, 0.97),
    # ContinuousParameter('tr(tm)', 1.05, 5.0),
])

# Exact linear constraints of the form: b_low < AX < b_up
A = np.array([1, -1, 0])[:,None].T 
b_up = np.array([np.inf]) 
b_low = np.array([0.03]) 
# space.constraints = [LinearInequalityConstraint(A, b_low, b_up)]


## Data transformations:

# Parameter transformation: 
from parameter_transform import quadratic
X_transform = None
# X_transform = quadratic((0.0054117094, 0.0426763236, 0.003307491))

# Cost function output transformation: 
Y_transform = None
# Y_transform.constants = (1, 1)


## GP model specification:
# These are hard-coded in bo_model(), see ./tools.py.


## Optimization loop controls:
n_iterations = 5             #optimisation iterations 
resume = False                  #resume from data file
training_data_file = "data.csv" #training data file
n_initial_data_points = 5         #exploration via Latin sampling (only if resume=False)


## Definition of external function evaluation:

# from dns import dns
# run_simulation.external_solver_interface = dns #interface to solver, see dns.py  

from dns import branin
external_solver_interface = branin   # the Branin test function


## Interface with DNS for post-processing:
path_solver = './dns_cases/'
init_stats=39999                #initial time step for statistics
W = 0.26 
p0 = 0.00240474482963
Ub = 2./3.
Re = 4200.
D = 5.07
n_discs = 16