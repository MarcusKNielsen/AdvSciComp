import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

# Define parameters
Lx = 80        # Spatial domain [-Lx, Lx]
Nx = 512       # Number of spatial points
x = np.linspace(-Lx, Lx, Nx)  # Spatial grid
dx = x[1] - x[0]              # Spatial step size
dt = 0.1                      # Time step size
t_max = 120                   # Maximum time for simulation
Nt = int(t_max / dt)          # Number of time steps

# Define soliton function
def soliton(x, x0, c):
    return 0.5 * c * (1 / np.cosh(0.5 * np.sqrt(c) * (x - x0))) ** 2

# Initial condition: sum of two solitons
u0 = soliton(x, -40, 0.5) + soliton(x, -15, 0.25)

# Fourier spectral method parameters
k = fftfreq(Nx, d=dx) * 2 * np.pi  # Wavenumbers

# Time evolution using 4th-order Runge-Kutta in Fourier space
def rk4_kdv(u0, k, dt, Nt):
    u = u0.copy()
    u_hat = fft(u)
    sol = [u.copy()]
    
    for _ in range(Nt):
        N_hat = lambda u_hat: -1j * k * fft(ifft(u_hat)**2)
        
        u1_hat = u_hat + 0.5 * dt * (-1j * k**3 * u_hat + N_hat(u_hat))
        u2_hat = u_hat + 0.5 * dt * (-1j * k**3 * u1_hat + N_hat(u1_hat))
        u3_hat = u_hat + dt * (-1j * k**3 * u2_hat + N_hat(u2_hat))
        u_hat = u_hat + (dt / 6) * (-1j * k**3 * (u_hat + 2 * u1_hat + 2 * u2_hat + u3_hat)
                                    + N_hat(u_hat) + 2 * N_hat(u1_hat) + 2 * N_hat(u2_hat) + N_hat(u3_hat))
        
        u = np.real(ifft(u_hat))
        sol.append(u)
    
    return np.array(sol)

# Run the simulation
u_sol = rk4_kdv(u0, k, dt, Nt)

# Plotting
time = np.arange(0, t_max + dt, dt)
extent = [-Lx, Lx, t_max, 0]

plt.figure(figsize=(10, 6))
plt.imshow(u_sol, extent=extent, aspect='auto', cmap='viridis')
plt.colorbar(label='u(x, t)')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Space-Time Plot of Soliton Collision (KdV Equation)')
plt.show()
