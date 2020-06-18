import numpy as np
import matplotlib.pyplot as plt
from function_definitions import *

H = -1; # depth of water (1 = non-dimensional)
K = 20; # number of grid points
grid = np.linspace(0,H,K) # grid array
kappa = 0.1*np.ones(grid.shape) # diffusivity array
dz = H/(K-1) # calculate grid spacing
N0 = 1 # bottom boundary condition on nutrient
P0 = 0 # bottom boundary condition on phytoplankton
N = np.zeros(grid.shape) # initial nutrient concentration
P = np.ones(grid.shape) # initial phytoplankton concentration
G = 1*np.exp(grid) # phytoplankton growth rate
k = 1 # half-saturation constant
m = 0.01 # phytoplankton mortality rate

## set up matrix equations AN*N = bN and AP*P = bP
AN,bN = diffusion_finite_difference(kappa,dz,N0)
AP,bP = diffusion_finite_difference(kappa,dz,P0)
## add biological terms to matrix equations
AN = AN + np.diag(-G*P/(N+k))
bN = bN - m*P
AP = AP + np.diag(G*N/(N+k))
bP = bP + m*P
## solve equations
N = np.linalg.solve(AN,bN)
P = np.linalg.solve(AP,bP)
