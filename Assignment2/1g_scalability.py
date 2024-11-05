#%% Importing modules
import numpy as np
import sys
sys.path.insert(0,r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\2_semester\Advanced nummerical\AdvSciComp\Assignment2\func")
from fourier import nodes, diff_matrix
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
#import Assignment2.functions_TIME as functions
import functions_TIME as functions
from L2space import discrete_L2_norm
from time import perf_counter

#%% Convergence test: 
# Set up different values for N (number of grid points) 
N_max = 500
N_values = np.arange(10,N_max,10) #N_max//10
errors_alias = []
errors_dealias = []
time_alias = []
time_dealias = []

# Constants 
c_value = 0.25
tf =  1.0
alpha = 0.5
w0 = np.pi

for N in N_values:

    x1, x2 = 40, 40

    # Fine grid (zero-padding) 
    M = 3*N//2 

    # Grid points and differentiation matrices
    w = nodes(N)
    D = diff_matrix(N)
    D3 = D @ D @ D
    a = 2 * np.pi / (x1 + x2)
    
    # Time integration
    max_step = alpha * 1.73 * 8 / (N**3 * a**3)

    # Initial condition dealiasing
    t = perf_counter()
    x = w*(x1+x2)/(2*np.pi) - x1 
    u0 = functions.dealias_IC(N,M,w0,x1,x2,c_value)
    sol_dealias = solve_ivp(functions.f_alias_free,[0, tf],u0,args=(D,D3,a,N,M),max_step=max_step,dense_output=True,method="RK23")
    time_dealias.append(perf_counter()-t)

    t = perf_counter()
    x = w*(x1+x2)/(2*np.pi) - x1 
    x0 = w0*(x1+x2)/(2*np.pi) - x1
    u0 = functions.u_exact(x, 0, c_value, x0)
    sol_alias = solve_ivp(functions.f, [0, tf], u0, args=(D, D3,a), max_step=max_step,dense_output=True, method="RK23")
    time_alias.append(perf_counter()-t)

    # Extract solution at final time  
    U_approx_dealias = sol_dealias.y[:, -1]
    U_approx_alias = sol_alias.y[:, -1]

    # Exact solution at final time
    x  = w*(x1+x2)/(2*np.pi) - x1
    x0 = w0*(x1+x2)/(2*np.pi) - x1
    U_exact = functions.u_exact(x, tf, c_value, x0)

    # Compute the L2 error
    err_dealias = U_approx_dealias-U_exact
    err_alias = U_approx_alias-U_exact
    weights_dealias = np.ones_like(err_dealias)*2*np.pi/N
    weights_alias = np.ones_like(err_alias)*2*np.pi/N
    error_dealias = discrete_L2_norm(err_dealias,weights_dealias)
    error_alias = discrete_L2_norm(err_alias,weights_alias)
    errors_dealias.append(error_dealias)
    errors_alias.append(error_alias)

plt.figure()
plt.semilogy(N_values,errors_dealias,".-",label=r"de-alias$")
plt.semilogy(N_values,errors_alias,".-",label=r"alias$")
plt.xlabel(r"$N$")
plt.ylabel(r"$e=\Vert u_N - u \Vert_{L^2}$")
plt.title("Convergence plot of solver")
plt.legend()
plt.figure()
plt.semilogy(N_values,time_dealias,".-",label=r"de-alias")
plt.semilogy(N_values,time_alias,".-",label=r"alias")
plt.xlabel(r"$N$")
plt.ylabel(r"time (s)")
plt.title("Scalability analysis of solver")
plt.legend()

plt.show()





