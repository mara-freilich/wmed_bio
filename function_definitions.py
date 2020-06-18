import numpy as np

def diffusion_finite_difference(kappa,dz,N0):
    '''Set up the arrays for 1D finite difference solver
        Uses a Dirichlet bottom boundary condition and a Neumann surface
        condition (to model nutrients entering from ocean bottom)
        
        Inputs:
        kappa:           float: diffusivity
        dz:              float: grid spacing
        N0:              float: bottom boundary value
        
        Output:
        A:      numpy array: left-hand side
        b:      numpy array: right-hand side
        '''
    
    d1 = -(kappa[:-1] + kappa[1:]) # diagonal entries
    d1 = np.append(-kappa[0],d1)
    
    A = (1/dz**2)*(np.diag(d1)+np.diag(kappa[:-1],1)+np.diag(kappa[:-1],-1))
    b = np.zeros(kappa.shape)
    b[-1] = b[-1]-N0*(kappa[-1]/dz**2)
    
    return A, b

def diffusion_time_stepping(kappa,dz,dt,N0,N):
    '''Solve diffusion equation in 1D using forward Euler time stepping
    
    Inputs:
    kappa:           float: diffusivity
    dz:              float: grid spacing
    dt:              float: time step
    N0:              float: bottom boundary value of nutrient
    N:               float: array of values at current time-step
    
    Output:
    Nj:              float: array of values at next time-step
    '''
    Nj = N
    d = kappa*dt/dz**2
    # Compute arrays for diffusion
    Njp1 = np.append(Nj[1:],N0) # Dirichlet boundary condition (fixed value)
    Njm1 = np.append(Nj[0],Nj[:-1]) # Neumann boundary condition (no-flux)
    # Time stepping
    Nj = d*Njp1+(1-2*d)*Nj+d*Njm1
    
    return Nj
