import numpy as np
import func.legendre as legendre
from advection_v2 import f_func, g_func, total_grid_points
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
x_left = -1
x_right = 1
a = 1.0 
alpha = 0.0
max_step = 0.001
tf = 0.1
formulation = "w"

def g0_val(t):
    return np.sin(np.pi*(-1 - a * t))

# Convergence test
N_list = [2,4,6,8]  # Example N values for rows
K_list = [2, 4, 8, 16, 32, 64]  # Example K values for columns

error = np.zeros((len(N_list), len(K_list)))
#time_matrix = 

for N_idx, N in enumerate(N_list):
    for K_idx, K in enumerate(K_list):

        x_nodes = legendre.nodes(N)
        x_total = total_grid_points(K, x_nodes, x_left, x_right)
        u0 = np.sin(np.pi * x_total)
        h = (x_total[-1] - x_total[0]) / K
        
        V, Vx, w = legendre.vander(x_nodes)
        M = np.linalg.inv(V @ V.T)
        Mk = (h / 2) * M
        Mk_inv = np.linalg.inv(Mk) 
        Dx = Vx @ np.linalg.inv(V)
        S = M @ Dx 

        sol = solve_ivp(f_func, [0, tf], u0, args=(Mk_inv, S, N, alpha, a, g0_val, formulation), max_step=max_step, dense_output=True, method="Radau")

        err = sol.y[:,-1] - g_func(x_total, sol.t[-1], a)
        I = np.eye(K)
        M_total = np.kron(I,Mk)
        error[N_idx,K_idx] = np.sqrt(err @ M_total @ err)    

# Calculate convergence rates (log-log slope)
convergence_rate = np.zeros(len(N_list))

for N_idx,N in enumerate(N_list):
    # Fit a first-order polynomial (line)
    coefficients = np.polyfit(np.log(K_list), np.log(error[N_idx]), 1)

    # Extract slope (a) and intercept (b)
    a, b = coefficients

    convergence_rate[N_idx] = a

# Display results as a table
print("N \\ K", *K_list, "Convergence rate", sep="\t")
for N_idx, N in enumerate(N_list):
    row = [f"{error[N_idx, K_idx]:.1E}" if error[N_idx, K_idx] != 0 else "-" for K_idx in range(len(K_list))]
    rate = f"{convergence_rate[N_idx]:.1f}"
    print(f"{N}\t" + "\t".join(row) + f"\t{rate}")


plt.figure()
plt.imshow(error)
plt.show()

