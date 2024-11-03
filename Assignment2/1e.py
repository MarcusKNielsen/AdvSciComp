#%% Importing modules
import numpy as np
import sys
sys.path.insert(0, r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\2_semester\Advanced nummerical\AdvSciComp\Assignment2\func")
from fourier import nodes, diff_matrix
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fft import fft
import functions_TIME as functions

dealias = False

#%% e) Aliasing errors
N = 50
M = 3*N//2  # Fine grid (zero-padding)
w = nodes(N)
D = diff_matrix(N)
D3 = D @ D @ D
x1, x2 = 20, 20
a = 2 * np.pi / (x1 + x2)
alpha = 0.01
max_step = alpha * 1.73*8/(N**3*a**3)
N2 = 300
w_lin = nodes(N2)  # Linear grid for exact solution
N1 = N
w0 = np.pi
x_lin = w_lin*(x1 + x2) / (2 * np.pi) - x1

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
    
    tf = 10.0

    # Solve with or without dealiasing
    if dealias:
        sol = solve_ivp(functions.f_alias_free, [0, tf], u0, args=(D, D3, a, N, M), max_step=max_step, dense_output=True, method="RK23")
    else:
        sol = solve_ivp(functions.f, [0, tf], u0, args=(D, D3,a), max_step=max_step, dense_output=True, method="RK23")

    # Initialize arrays to store frequencies and Fourier transform magnitudes for both solutions
    uk_approx_magnitudes = np.zeros((len(sol.t), N1))
    uk_exact_magnitudes  = np.zeros((len(sol.t), N1))

    # Compute frequencies
    #dw1 = (w[1:] - w[:-1])[0]
    #freq1 = scipy.fft.fftfreq(N1, d=dw1)
    #dw2 = (w_lin[1:] - w_lin[:-1])[0]
    #freq2 = scipy.fft.fftfreq(N2, d=dw2)
    freq_sorted = np.arange(-N1//2,N1//2)
    
    # Populate the Fourier coefficients of the approximate solution over time
    for idx_time, time in enumerate(sol.t):
        
        U_approx  = sol.y[:, idx_time]
        uk_approx = fft(U_approx)
        uk_approx = np.concatenate([uk_approx[N1-N1//2:],uk_approx[:N1//2]])

        # Store the magnitudes of the Fourier transform for the approximate solution
        uk_approx_magnitudes[idx_time, :] = np.abs(uk_approx)
        
        # Calculate the Fourier transform of the exact solution on the finer grid (N2)
        uk_exact_full = fft(functions.u_exact(x_lin, time, c_i, x0))
        uk_exact_full = (N1/N2)*np.concatenate([uk_exact_full[N2-N1//2:],uk_exact_full[:N1//2]])
        uk_exact_magnitudes[idx_time, :] = np.abs(uk_exact_full)
        

    # Correct the code for the exact Fourier coefficients mesh plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Mesh plot for the Fourier transform of the approximate solution
    T,K = np.meshgrid(sol.t,freq_sorted)
    mesh1 = ax1.pcolormesh(T, K, uk_approx_magnitudes.T, shading='auto', cmap='viridis')
    fig.colorbar(mesh1, ax=ax1, label=r"$|u_k|$ (Approximate)")
    ax1.set_ylabel(r"$k$")
    ax1.set_xlabel("Time")
    ax1.set_title(f"Approximate Fourier Transform Magnitude Over Time for c={c_i}")

    # Mesh plot for the Fourier transform of the exact solution on the finer grid
    T,K = np.meshgrid(sol.t,freq_sorted)
    mesh2 = ax2.pcolormesh(T, K, uk_exact_magnitudes.T, shading='auto', cmap='viridis')
    fig.colorbar(mesh2, ax=ax2, label=r"$|u_k|$ (Exact)")
    ax2.set_ylabel(r"$k$")
    ax2.set_xlabel("Time")
    ax2.set_title(f"Exact Fourier Transform Magnitude Over Time for c={c_i}")

    plt.tight_layout()


plt.show()

