#%% modules
import pandas as pd
import numpy as np
import func.L2space as L2space
import func.legendre as legendre
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy
import advection_v2 

#%% Functions
def set_up_A_mat_advection(x_nodes,h,a):
    
    V,Vx,_ = legendre.vander(x_nodes)
    M = np.linalg.inv(V@V.T)
    Mk = (h/2)*M
    Mk_inv = np.linalg.inv(Mk) 
    Dx = Vx@np.linalg.inv(V)
    S = M@Dx

    A = Mk_inv@(((S.T*a)))  

    return A

#%% Initializations

# Does not change
x_left = -1
x_right = 1
a = 1
alpha = 0 # upwind alpha = 0, central alpha = 1

#%% Eigen values test

# looping values
N_list = np.arange(10,100,10)
number_element_list =np.arange(10,100,10)

Eig_mat = np.zeros([len(N_list),len(number_element_list)])

for idx_ni,ni in enumerate(N_list):

    x_nodes = legendre.nodes(ni)
    
    for idx_e,e in enumerate(number_element_list):

        x_total = advection_v2.total_grid_points(e,x_nodes,x_left,x_right)

        h = (x_total[-1]-x_total[0])/e

        A = set_up_A_mat_advection(x_nodes,h,a)
        Eig_mat[idx_ni,idx_e] = np.max(np.abs(np.linalg.eigvals(A)))


plt.figure()
N, K = np.meshgrid(N_list, number_element_list)

# Create the pcolormesh plot
pcm = plt.pcolormesh(K, N, Eig_mat.T)

# Label the axes and add a title
plt.xlabel("k: number of elements")
plt.ylabel("N: number of nodes")
plt.title("maximum eigenvalues")

# Add the colorbar
plt.colorbar(pcm, label="max|$\lambda$|")


plt.figure()
plt.title("Convergence Test")

for k_idx,K in enumerate(number_element_list):
    
    plt.loglog(N_list,Eig_mat[:,k_idx],"-o",label=f"K={K}")

    # Fit a first-order polynomial (line)
    coefficients = np.polyfit(np.log(N_list), np.log(Eig_mat[:,k_idx]), 1)

    # Extract slope (a) and intercept (b)
    a, b = coefficients
    
    print(f"K = {K}")
    print(f"a = {a}")
    #print(f"Intercept (b): {b}")

plt.xlabel(r"$N$: Number of Elements")
plt.ylabel(r"$\max|u - u_h|$")
plt.legend()




# Assuming Eig_mat, N_list, and number_element_list are defined
N, K = np.meshgrid(N_list, number_element_list)

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the surface plot
surf = ax.plot_surface(K, N, Eig_mat.T, cmap="viridis")

# Label the axes and add a title
ax.set_xlabel("k: number of elements")
ax.set_ylabel("N: number of nodes")
ax.set_zlabel("max|$\lambda$|")
ax.set_title("Maximum Eigenvalues")

# Add the colorbar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="max|$\lambda$|")

plt.show()

