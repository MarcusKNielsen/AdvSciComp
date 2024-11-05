#%% Importing modules
import numpy as np
import sys
sys.path.insert(0, r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\2_semester\Advanced nummerical\AdvSciComp\Assignment2\func")
from fourier import nodes, diff_matrix
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fft import fft
import scipy
import functions_TIME as functions
from matplotlib.colors import LogNorm

dealias = True

#%% e) Aliasing errors 
N = 30
M = 3*N//2  # Fine grid (zero-padding) 
w = nodes(N)
D = diff_matrix(N)
D3 = D @ D @ D
x1, x2 = 20, 20
a = 2 * np.pi / (x1 + x2)
alpha = 0.5
max_step = alpha * 1.73*8/(N**3*a**3)
N2 = 500
w_lin = nodes(N2)  # Linear grid for exact solution
N1 = N
w0 = np.pi
x_lin = w_lin*(x1 + x2) / (2 * np.pi) - x1


c = np.array([0.25, 0.5, 1])

# Define the figure with 3 rows and 2 columns for subplots
fig, axes = plt.subplots(3, 2, figsize=(8, 8))

fig.suptitle('Evolution of Fourier Coefficients over Time')

# Loop through each c_i and create the subplots
for idx, c_i in enumerate(c):
    
    # Set up initial conditions for dealiasing or normal computation
    if dealias:
        x = w*(x1 + x2) / (2 * np.pi) - x1
        x0 = w0*(x1 + x2) / (2 * np.pi) - x1
        u0 = functions.dealias_IC(N, M, w0, x1, x2, c_i)
    else:
        x = w*(x1 + x2) / (2 * np.pi) - x1
        x0 = w0*(x1 + x2) / (2 * np.pi) - x1
        u0 = functions.u_exact(x, 0, c_i, x0)
    
    tf = 10.0

    # Solve with or without dealiasing
    if dealias:
        sol = solve_ivp(functions.f_alias_free, [0, tf], u0, args=(D, D3, a, N, M), max_step=max_step, dense_output=True, method="RK23")
    else:
        sol = solve_ivp(functions.f, [0, tf], u0, args=(D, D3, a), max_step=max_step, dense_output=True, method="RK23")

    # Initialize arrays to store frequencies and Fourier transform magnitudes for both solutions
    uk_approx_magnitudes = np.zeros((len(sol.t), N1))
    uk_exact_magnitudes  = np.zeros((len(sol.t), N1))

    # Compute frequencies
    freq_sorted = np.arange(-N1//2, N1//2)
    
    # Populate the Fourier coefficients of the approximate solution over time
    for idx_time, time in enumerate(sol.t):
        
        U_approx  = sol.y[:, idx_time]
        uk_approx = fft(U_approx)
        uk_approx = np.concatenate([uk_approx[N1-N1//2:], uk_approx[:N1//2]])

        # Store the magnitudes of the Fourier transform for the approximate solution
        uk_approx_magnitudes[idx_time, :] = np.abs(uk_approx)
        
        # Calculate the Fourier transform of the exact solution on the finer grid (N2)
        uk_exact_full = fft(functions.u_exact(x_lin, time, c_i, x0))
        uk_exact_full = (N1/N2) * np.concatenate([uk_exact_full[N2-N1//2:], uk_exact_full[:N1//2]])
        uk_exact_magnitudes[idx_time, :] = np.abs(uk_exact_full)
        

    # Create mesh grids for time and frequency
    T, K = np.meshgrid(sol.t, freq_sorted)
    
    # Plot the Fourier transform of the approximate solution
    mesh1 = axes[idx, 0].pcolormesh(T, K, uk_approx_magnitudes.T, shading='auto', cmap='viridis', norm=LogNorm())
    fig.colorbar(mesh1, ax=axes[idx, 0], label=r"$|u_k|$ (Approximate)")
    axes[idx, 0].set_ylabel(r"$k$")
    axes[idx, 0].set_xlabel("Time")
    axes[idx, 0].set_title(f"Approximate Coefficients with c={c_i}")

    # Plot the Fourier transform of the exact solution
    mesh2 = axes[idx, 1].pcolormesh(T, K, uk_exact_magnitudes.T, shading='auto', cmap='viridis', norm=LogNorm())
    fig.colorbar(mesh2, ax=axes[idx, 1], label=r"$|u_k|$ (Exact)")
    axes[idx, 1].set_ylabel(r"$k$")
    axes[idx, 1].set_xlabel("Time")
    axes[idx, 1].set_title(f"Exact Coefficients with c={c_i}")

plt.subplots_adjust(hspace=0.9)

plt.tight_layout()
plt.show()




#%%

from L2space import discrete_L2_norm
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
N = 50
c_values = [0.25, 0.5, 1.0]
tf = 1.0
alpha = 0.5
x1, x2 = 40, 40
w0 = np.pi
M = 3 * N // 2

# Font size variables
title_fontsize = 10
label_fontsize = 9
tick_fontsize = 10

# Set up the main plot
plt.figure(dpi=150)
plt.gcf().set_size_inches(w=4.0, h=4.0)

N_max = 150
N_values = np.arange(10, N_max, 4)  # Range of N values for convergence testing

# Get the default color cycle from matplotlib
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Outer loop for dealiasing condition
for i, c_value in enumerate(c_values):
    color = color_cycle[i % len(color_cycle)]  # Cycle through default colors for each c_value

    for dealias in [False, True]:
        errors = []  # List to store errors for the current c_value and dealiasing condition

        for N in N_values:
            # Fine grid (zero-padding)
            M = 3 * N // 2

            # Grid points and differentiation matrices
            w = nodes(N)
            D = diff_matrix(N)
            D3 = D @ D @ D
            a = 2 * np.pi / (x1 + x2)

            # Initial condition
            x = w * (x1 + x2) / (2 * np.pi) - x1
            if dealias:
                u0 = functions.dealias_IC(N, M, w0, x1, x2, c_value)
            else:
                x0 = w0 * (x1 + x2) / (2 * np.pi) - x1
                u0 = functions.u_exact(x, 0, c_value, x0)

            # Time integration
            max_step = alpha * 1.73 * 8 / (N**3 * a**3)
            if dealias:
                sol = solve_ivp(functions.f_alias_free, [0, tf], u0, args=(D, D3, a, N, M), max_step=max_step, dense_output=True, method="RK23")
            else:
                sol = solve_ivp(functions.f, [0, tf], u0, args=(D, D3, a), max_step=max_step, dense_output=True, method="RK23")

            # Extract solution at final time
            U_approx = sol.y[:, -1]

            # Exact solution at final time
            x = w * (x1 + x2) / (2 * np.pi) - x1
            x0 = w0 * (x1 + x2) / (2 * np.pi) - x1
            U_exact = functions.u_exact(x, tf, c_value, x0)

            # Compute the L2 error
            err = U_approx - U_exact
            weights = np.ones_like(err) * 2 * np.pi / N
            error = discrete_L2_norm(err, weights)  # Use the discrete L2 norm
            errors.append(error)  # Append error for the current N

        # Determine line style and label
        linestyle = "--" if dealias else "-"
        label_type = "dealias" if dealias else "alias"
        label = fr"$c = {c_value}$, {label_type}"

        # Plot errors for the current c_value and dealiasing condition
        plt.semilogy(N_values, errors, linestyle=linestyle, color=color, marker=".", label=label)

# Customize the plot
plt.xlabel(r"$N$", fontsize=label_fontsize)
plt.ylabel(r"$\Vert u_N(x,T) - u(x,T) \Vert_{L^2}$", fontsize=label_fontsize)
plt.title("Convergence Plot", fontsize=title_fontsize)
plt.legend(fontsize=label_fontsize)
plt.tight_layout()

# Save the figure as a PNG file
plt.savefig("/home/max/Documents/DTU/AdvNumericalMethods/AdvSciComp/Assignment2/figures/1e_convergence_updated.png", format='png', bbox_inches="tight")
plt.show()








