#%% Importing modules
import numpy as np
import sys
sys.path.insert(0, r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\2_semester\Advanced nummerical\AdvSciComp\Assignment2\func")
from fourier import nodes, diff_matrix
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from L2space import discrete_L2_norm
import functions_TIME as functions
from matplotlib.colors import LogNorm

dealias = False

#%% Opgave d)
N_list = np.arange(10, 100, 2)  # Different grid resolutions
x_list = np.arange(10,  50, 2)  # Different domain sizes for x1 = x2 = x

c = np.array([0.25, 0.5, 1])  # Values of the parameter c
alpha = 0.5

# Setting up the subplots: 1 column and 3 rows
fig, axes = plt.subplots(3, 1, figsize=(8, 8), constrained_layout=True)

for idx, c_i in enumerate(c):
    # Initialize 2D array to store errors for each (N, x) pair
    errors = np.zeros([len(N_list), len(x_list)])

    # Loop over each value of N and x
    for i, N in enumerate(N_list):
        for j, x in enumerate(x_list):
            # Set x1 and x2 to the current x value from x_list
            x1 = x
            x2 = x

            # Fine grid (zero-padding)
            M = 3 * N // 2
            w = nodes(N)
            D = diff_matrix(N)
            D3 = D @ D @ D
            a = 2 * np.pi / (x1 + x2)
            w0 = np.pi 
            tf = 1.0
            max_step = alpha * 1.73 * 8 / (N**3 * a**3)

            # Define spatial points and initial condition for the current N and x
            x_vals = w * (x1 + x2) / (2 * np.pi) - x1
            x0 = w0 * (x1 + x2) / (2 * np.pi) - x1
            u0 = functions.u_exact(x_vals, 0, c_i, x0)

            # Solving the ODE
            if dealias:
                sol = solve_ivp(functions.f_alias_free, [0, tf], u0, args=(D, D3, a, N, M), max_step=max_step, dense_output=True, method="RK23")
            else:
                sol = solve_ivp(functions.f, [0, tf], u0, args=(D, D3, a), max_step=max_step, dense_output=True, method="RK23")

            # Compute the error between the solution and exact solution
            U_exact = functions.u_exact(x_vals, tf, c_i, x0)  # Exact solution at final time
            U_numerical = sol.y[:, -1]  # Numerical solution at final time
            err = U_numerical - U_exact
            weights = np.ones_like(err) * 2 * np.pi / N
            error = discrete_L2_norm(err, weights)

            # Store the error for this combination of N and x
            errors[i, j] = error
    
    # Plot in the corresponding subplot
    ax = axes[idx]
    cplot = ax.pcolormesh(N_list, x_list, errors.T, shading='auto', norm=LogNorm())
    cbar = fig.colorbar(cplot, ax=ax, label=r"$||u_N-u||_{L^2}$")
    cbar.set_label(r"$||u_N-u||_{L^2}$", fontsize=14) 
    ax.set_xlabel("N (grid points)", fontsize=14)
    ax.set_ylabel("x (domain size)", fontsize=14)
    ax.set_title(f"Error plot for c = {c_i}",fontsize=14)
    ax.plot(N_list,x_list[np.argmin(errors,axis=1)],"--",color="red",label=r"low error line")
    ax.legend(loc="upper left",fontsize=12)

plt.savefig("/home/max/Documents/DTU/AdvNumericalMethods/AdvSciComp/Assignment2/figures/1d_investigation.png", format='png', bbox_inches="tight")
plt.show()
