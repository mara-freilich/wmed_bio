import numpy as np
import matplotlib.pyplot as plt
from function_definitions import *

H = -1; # depth of water (1 = non-dimensional)
K = 20; # number of grid points
grid = np.linspace(0,H,K) # grid array
kappa = 0.01 # diffusivity value
dz = H/(K-1); # calculate grid spacing
N0 = 1; # bottom boundary condition on nutrient
N = np.zeros(grid.shape) # initial condition
Nsave = [N] # save initial condition
dt = 0.01 # time step
nt = 50 # number of time steps
time = np.arange(nt+1)*dt
for t in np.arange(nt):
    N = diffusion(kappa,dz,dt,N0,N)
    Nsave = np.concatenate((Nsave,[N]),axis = 0)

plt.pcolor(time,grid,Nsave.T)
plt.xlabel('time in days')
plt.ylabel('depth of water column')
