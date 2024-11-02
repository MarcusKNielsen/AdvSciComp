import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,r"/home/max/Documents/DTU/AdvNumericalMethods/AdvSciComp/Assignment2/func")
sys.path.insert(0,r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\2_semester\Advanced nummerical\AdvSciComp\Assignment2\func")
sys.path.append(r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\2_semester\Advanced nummerical\AdvSciComp\Assignment2")
from legendre import vander, nodes
from L2space import discrete_inner_product, discrete_L2_norm

def u_exact(x,epsilon):
    return (np.exp(-x / epsilon) + (x - 1) - np.exp(-1 / epsilon )*x) / (np.exp(-1 / epsilon) - 1)

def a_coefs(N,eps):

    n = np.arange(N)
    result = np.zeros_like(n, dtype=float)

    # p = 0
    result[n == 0] = 0

    # p even
    even_condition = (n % 2 == 0) & (n != 0)
    n_even = n[even_condition] # indices to multiply
    result[even_condition] = -2 * eps * n_even * (n_even + 1)

    # p odd
    odd_condition = (n % 2 != 0)
    result[odd_condition] = -2
    
    return result

def b_coefs(N,eps):

    p = np.arange(N)
    result = np.zeros_like(p, dtype=float)

    # p = 0,1
    result[p == 0] = 0
    result[p == 1] = 0

    # p even
    p_even = (p % 2 == 0) & (p != 0)
    result[p_even] = 1

    # p odd
    p_odd = (p % 2 != 0) & (p != 1)
    np_even = p[p_odd]
    result[p_odd] = eps * (np_even * (np_even + 1) - 2)
    
    return result

# parameters
N = 100
eps = 0.001

def solve_tau(N,eps):

    x = nodes(N)
    V,Vx,w = vander(x,Normalize=False)
    
    # Setup A matrix
    A = np.zeros([N,N])
    
    # The n=0 equation
    A[0] = a_coefs(N,eps)
    
    # The n=1 equation
    A[1] = b_coefs(N,eps)
    
    # three term part
    for n in range(2,N-2):
        A[n,n-1] = 1/(2*eps*(2*n-1))
        A[n,n]   = 1
        A[n,n+1] = -1/(2*eps*(2*n+3))
    
    # boundary condition
    A[-2] = V[0]
    A[-1] = V[-1]
    
    # Compute hat_f_0
    hat_f_0 = 1
    
    # Setup right hand side
    b = np.zeros(N)
    b[0] =  hat_f_0
    b[2] = -hat_f_0/(12*eps)
    
    # solve for coefficients
    u_hat = solve(A,b)
    
    return x, V@u_hat, w

# x_lin = np.linspace(-1,1,1000)
# # plot solution
# plt.plot(x,V@u_hat,".-",label="approx")
# plt.plot(x_lin,u_exact((x_lin+1)/2,eps),label="exact")
# plt.legend()
# plt.show()

#%%

# Set font size, tick size, and dot size
font_size = 16
tick_size = 14
dot_size = 12

N = 32
epsilon_values = np.array([0.1, 0.01, 0.001])

# Prepare the figure with a 2x2 grid of subplots
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
x_lin = np.linspace(-1, 1, 1000)

# Plot the solutions for each epsilon on the first three subplots
for idx, eps in enumerate(epsilon_values):
    row, col = divmod(idx, 2)
    ax[row, col].plot((x_lin+1)/2, u_exact((x_lin+1)/2, eps), label="Exact")
    x,u,_ = solve_tau(N,eps)
    ax[row, col].plot((x+1)/2, u, ".", label=rf"Numerical", markersize=dot_size)

    # Customize the subplot with font size and tick size
    ax[row, col].set_title(rf"Solution for $\varepsilon$={eps} with N={N}", fontsize=font_size)
    ax[row, col].legend(fontsize=font_size)
    ax[row, col].set_xlabel('x', fontsize=font_size)
    ax[row, col].set_ylabel('u(x)', fontsize=font_size)
    ax[row, col].tick_params(axis='both', which='major', labelsize=tick_size)

# Convergence plot in the last subplot
N_array = np.logspace(np.log10(4), np.log10(200), num=50, dtype=int)
N_array = np.unique(N_array)

error_temp = np.zeros([len(N_array), len(epsilon_values)])

for N_idx, N in enumerate(N_array):
    
    # Case 1
    x,u,w = solve_tau(N,epsilon_values[0])
    u_ex = u_exact((x+1)/2, epsilon_values[0])
    error_temp[N_idx,0] = discrete_L2_norm(u-u_ex,w)
    
    # Case 2
    x,u,w = solve_tau(N,epsilon_values[1])
    u_ex = u_exact((x+1)/2, epsilon_values[1])
    error_temp[N_idx,1] = discrete_L2_norm(u-u_ex,w)
    
    # Case 3
    x,u,w = solve_tau(N,epsilon_values[2])
    u_ex = u_exact((x+1)/2, epsilon_values[2])
    error_temp[N_idx,2] = discrete_L2_norm(u-u_ex,w)


# Plot the convergence in the last subplot (ax[1,1])
ax[1, 1].set_title("Convergence Plot", fontsize=font_size)
for i in range(3):
    ax[1, 1].semilogy(N_array, error_temp[:, i], ".-", label=rf"$\epsilon$={epsilon_values[i]}", markersize=dot_size)
ax[1, 1].legend(fontsize=font_size)
ax[1, 1].set_xlabel('N', fontsize=font_size)
ax[1, 1].set_ylabel(r'$\Vert u - u_N \Vert_{L^2}$', fontsize=font_size)
ax[1, 1].tick_params(axis='both', which='major', labelsize=tick_size)

# Adjust layout to prevent overlap
plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=2.5)

# Show the plots
plt.show()


