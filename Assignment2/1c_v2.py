#%% Importing modules
import numpy as np
import sys
sys.path.insert(0, r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\2_semester\Advanced nummerical\AdvSciComp\Assignment2\func")
from fourier import nodes, diff_matrix
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from L2space import discrete_inner_product
#import Assignment2.functions_TIME as functions
import functions_TIME as functions

dealias = True

#%% Opgave d)
N = 200
# Fine grid (zero-padding)
M = 3 * N // 2
w = nodes(N)
D = diff_matrix(N)
D3 = D @ D @ D
x1, x2 = 40, 40
a = 2 * np.pi / (x1 + x2)
w0 = np.pi 
tf = 20.0

c = np.array([0.25, 0.5, 1])

alpha = 0.5
max_step = alpha * 1.73 * 8 / (N**3 * a**3)

# Lists to store results for each c
int_M_test = []
int_V_test = []
int_E_test = []
t_values = []  # Store time arrays for each c

# Create a 3x2 subplot for L2 and Linf errors
fig, axes = plt.subplots(3, 2, dpi=100, figsize=(10, 8))  # Adjust figsize as needed

fontsize = 13

for idx, c_i in enumerate(c):
    if dealias:
        x = w * (x1 + x2) / (2 * np.pi) - x1 
        x0 = w0 * (x1 + x2) / (2 * np.pi) - x1
        u0 = functions.dealias_IC(N, M, w0, x1, x2, c_i) 
    else:
        x = w * (x1 + x2) / (2 * np.pi) - x1 
        x0 = w0 * (x1 + x2) / (2 * np.pi) - x1
        u0 = functions.u_exact(x, 0, c_i, x0)

    if dealias:
        sol = solve_ivp(functions.f_alias_free, [0, tf], u0, args=(D, D3, a, N, M), max_step=max_step, dense_output=True, method="RK23")
    else:
        sol = solve_ivp(functions.f, [0, tf], u0, args=(D, D3, a), max_step=max_step, dense_output=True, method="RK23")

    t = sol.t
    U = sol.y.T
    t_values.append(t)  # Save t for this value of c

    # Initialize arrays for int_M_test, int_V_test, and int_E_test based on length of t
    int_M = np.zeros(len(t))
    int_V = np.zeros(len(t))
    int_E = np.zeros(len(t))
    L2_error = np.zeros(len(t))   # L2-norm error over time
    Linf_error = np.zeros(len(t)) # Linf-norm error over time

    for i in range(len(t)):
        weight = np.ones_like(x) * 2 * np.pi / N
        int_M[i] = discrete_inner_product(np.ones_like(U[i]), U[i], weight)
        int_V[i] = discrete_inner_product(U[i], U[i], weight)
        int_E_term1 = discrete_inner_product(D @ U[i], D @ U[i], weight)
        int_E_term2 = discrete_inner_product(U[i], U[i] * U[i], weight)
        int_E[i] = 0.5 * int_E_term1 - int_E_term2

        # Numerical solution at time t[i]
        U_approx = U[i]
        
        # Exact solution at time t[i]
        x  = w * (x1 + x2) / (2 * np.pi) - x1
        x0 = w0 * (x1 + x2) / (2 * np.pi) - x1
        U_exact = functions.u_exact(x, t[i], c_i, x0)
        
        # Compute L2 norm of the error
        L2_error[i] = np.sqrt(discrete_inner_product(U_approx - U_exact, U_approx - U_exact, weight))
        
        # Compute Linf norm of the error (max absolute error)
        Linf_error[i] = np.max(np.abs(U_approx - U_exact))

    # Plot L2 error in the first column of each row
    axes[idx, 0].plot(t, L2_error, label=f"c = {c_i}")
    axes[idx, 0].set_xlabel('Time (t)',fontsize=fontsize)
    axes[idx, 0].set_ylabel(r"$L^2$-error",fontsize=fontsize)
    axes[idx, 0].legend(loc='upper left',fontsize=fontsize)
    axes[idx, 0].grid(True)

    # Plot Linf error in the second column of each row
    axes[idx, 1].plot(t, Linf_error, linestyle='--', label=f"c = {c_i}")
    axes[idx, 1].set_xlabel('Time (t)',fontsize=fontsize)
    axes[idx, 1].set_ylabel(r'$L^\infty$-Error',fontsize=fontsize)
    axes[idx, 1].legend(loc='upper left',fontsize=fontsize)
    axes[idx, 1].grid(True)

    # Append results to lists
    int_M_test.append(int_M)
    int_V_test.append(int_V)
    int_E_test.append(int_E)

# Adjust layout and save figure
fig.suptitle(r'$L^2$ and $L^\infty$ Errors Over Time for Different Values of $c$', y=0.92)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("/home/max/Documents/DTU/AdvNumericalMethods/AdvSciComp/Assignment2/figures/1d_L2_Linf_errors.png", format='png', bbox_inches="tight")
plt.show()


# Second figure with 9 subplots (3x3 grid) for int_M_test, int_V_test, and int_E_test at each value of c
fig, axes = plt.subplots(3, 3, figsize=(8.27, 11.69), constrained_layout=True)  # A4 size in inches

for idx, c_i in enumerate(c):
    # Plot int_M_test
    axes[0, idx].plot(t_values[idx], int_M_test[idx], label=f"c = {c_i}")
    axes[0, idx].set_xlabel('Time (t)')
    axes[0, idx].set_ylabel(r'$M= \int u dx$')
    axes[0, idx].legend(fontsize=12, loc="upper right")
    axes[0, idx].grid(True)

    # Plot int_V_test
    axes[1, idx].plot(t_values[idx], int_V_test[idx], label=f"c = {c_i}")
    axes[1, idx].set_xlabel('Time (t)')
    axes[1, idx].set_ylabel(r'$V= \int u^2 dx$')
    axes[1, idx].legend(fontsize=12, loc="upper right")
    axes[1, idx].grid(True)

    # Plot int_E_test
    axes[2, idx].plot(t_values[idx], int_E_test[idx], label=f"c = {c_i}")
    axes[2, idx].set_xlabel('Time (t)')
    axes[2, idx].set_ylabel(r'$E= \int \frac{1}{2}u_x^2 - u^3 dx$')
    axes[2, idx].legend(fontsize=12, loc="lower right")
    axes[2, idx].grid(True)

# Save the 3x3 subplot figure
plt.savefig("/home/max/Documents/DTU/AdvNumericalMethods/AdvSciComp/Assignment2/figures/1d_Mass_Momentum_Energy.png", format='png', bbox_inches="tight")
plt.show()
