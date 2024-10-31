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

dealias = True

#%% e) Aliasing errors
N = 40
M = 3*N//2  # Fine grid (zero-padding)
w = nodes(N)
D = diff_matrix(N)
D3 = D @ D @ D
x1, x2 = 20, 20
a = 2 * np.pi / (x1 + x2)
alpha = 0.01
max_step = alpha * 1.73*8/(N**3*a**3)
x_lin = nodes(60)  # Linear grid for exact solution
N1 = N
N2 = len(x_lin)
w0 = np.pi

c = np.array([0.25, 0.5, 1])

for c_i in c:

    # Set up initial conditions for dealiasing or normal computation
    if dealias:
        x = w*(x1 + x2) / (2 * np.pi) - x1
        x0 = w0*(x1 + x2) / (2 * np.pi) - x1
        u0 = functions.dealias_IC(N, M, w0, x1, x2, c_i)
    else:
        x = w*(x1 + x2) / (2 * np.pi) - x1 
        x0 = w0*(x1 + x2) / (2 * np.pi) - x1
        u0 = functions.u_exact(x, 0, c_i, x0)

    tf = 1.0

    # Solve with or without dealiasing
    if dealias:
        sol = solve_ivp(functions.f_alias_free, [0, tf], u0, args=(D, D3, a, N, M), max_step=max_step, dense_output=True, method="RK23")
    else:
        sol = solve_ivp(functions.f, [0, tf], u0, args=(D, D3), max_step=max_step, dense_output=True, method="RK23")

    # Initialize arrays to store frequencies and Fourier transform magnitudes for both solutions
    uk_approx_magnitudes = np.zeros((len(sol.t), N1))
    uk_exact_magnitudes = np.zeros((len(sol.t), N2))

    # Compute frequencies
    dx1 = (x[1:] - x[:-1])[0]
    freq1 = scipy.fft.fftfreq(N1, d=dx1)
    dx2 = (x_lin[1:] - x_lin[:-1])[0]
    freq2 = scipy.fft.fftfreq(N2, d=dx2)

    # Calculate the Fourier transform of the exact solution on the finer grid (N2)
    uk_exact_full = fft(functions.u_exact(x_lin, tf, c_i, x0))

    # Store exact Fourier transform magnitudes (static across all time steps)
    uk_exact_magnitudes[:, :] = np.abs(uk_exact_full.real)

    # Populate the Fourier coefficients of the approximate solution over time
    for idx_time, time in enumerate(sol.t):
        U_approx = sol.y[:, idx_time]
        uk_approx = fft(U_approx)

        # Store the magnitudes of the Fourier transform for the approximate solution
        uk_approx_magnitudes[idx_time, :] = np.abs(uk_approx.real)

    # Correct the code for the exact Fourier coefficients mesh plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Mesh plot for the Fourier transform of the approximate solution 
    mesh1 = ax1.pcolormesh(sol.t, freq1, uk_approx_magnitudes.T, shading='auto', cmap='viridis')
    fig.colorbar(mesh1, ax=ax1, label=r"$|u_k|$ (Approximate)")
    ax1.set_ylabel(r"$k$")
    ax1.set_xlabel("Time")
    ax1.set_title(f"Approximate Fourier Transform Magnitude Over Time for c={c_i}")

    # Mesh plot for the Fourier transform of the exact solution on the finer grid 
    exact_magnitude_mesh = np.abs(uk_exact_full).reshape(N2, 1) @ np.ones((1, len(sol.t)))
    mesh2 = ax2.pcolormesh(sol.t, freq2, exact_magnitude_mesh.T, shading='auto', cmap='plasma')
    fig.colorbar(mesh2, ax=ax2, label=r"$|u_k|$ (Exact)")
    ax2.set_ylabel(r"$k$")
    ax2.set_xlabel("Time")
    ax2.set_title(f"Exact Fourier Transform Magnitude Over Time for c={c_i}")

    plt.tight_layout()


plt.show()

