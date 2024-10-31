#%% Importing modules
import numpy as np
import sys
sys.path.insert(0,r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\2_semester\Advanced nummerical\AdvSciComp\Assignment2\func")
from fourier import nodes, diff_matrix
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import Assignment2.functions_TIME as functions

dealias = True

#%% Convergence test: 
# Set up different values for N (number of grid points) 
N_values = np.arange(4,50,6)
errors = []

# Constants
c_value = 1.0
tf = 1.0
alpha = 0.01
x1, x2 = 20, 20
w0 = np.pi

for N in N_values:

    # Fine grid (zero-padding) 
    M = 3*N//2 

    # Grid points and differentiation matrices
    w = nodes(N)
    D = diff_matrix(N)
    D3 = D @ D @ D
    a = 2 * np.pi / (x1 + x2)

    # Initial condition dealiasing
    if dealias:
        x = w*(x1+x2)/(2*np.pi) - x1 
        u0 = functions.dealias_IC(N,M,w0,x1,x2,c_value)
    else:
        x = w*(x1+x2)/(2*np.pi) - x1 
        x0 = w0*(x1+x2)/(2*np.pi) - x1
        u0 = functions.u_exact(x, 0, c_value, x0)
    
    # Time integration
    max_step = 1e-10#alpha * 1.73 * 8 / (N**3 * a**3)

    if dealias:
        sol = solve_ivp(functions.f_alias_free,[0, tf],u0,args=(D,D3,a,N,M),max_step=max_step,dense_output=True,method="RK23")
    else:
        sol = solve_ivp(functions.f, [0, tf], u0, args=(D, D3), max_step=max_step,dense_output=True, method="RK23")

    # Extract solution at final time 
    U_approx = sol.y[:, -1]

    # Exact solution at final time
    x  = w*(x1+x2)/(2*np.pi) - x1
    x0 = w0*(x1+x2)/(2*np.pi) - x1
    U_exact = functions.u_exact(x, tf, c_value, x0)

    # Compute the L2 error
    error = np.max(np.abs(U_approx-U_exact))
    errors.append(error)

plt.figure()
plt.loglog(N_values,errors)
plt.xticks(np.arange(4,50,6))
plt.yticks([1e-1,1e-3,1e-5,1e-7,1e-10])
plt.minorticks_on()
plt.show()
