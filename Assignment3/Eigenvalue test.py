#%% modules
import pandas as pd
import numpy as np
import func.L2space as L2space
import func.legendre as legendre
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy
import advection_matrix as ad_mat

#%% Functions
def set_up_A_mat(x_nodes,h):
    
    V,Vx,_ = legendre.vander(x_nodes)
    M = np.linalg.inv(V@V.T)
    Mk = (h/2)*M
    Mk_inv = np.linalg.inv(Mk) 
    Dx = Vx@np.linalg.inv(V)
    S = M@Dx

    return Mk_inv,S

#%% Initializations

# Does not change
x_left = -1
x_right = 1
a = 1
alpha = 0 # upwind alpha = 0, central alpha = 1

#%% Eigen values test

# looping values
N_list = np.arange(3,30,3)
number_element_list =np.arange(3,20,3)

Eig_mat = np.zeros([len(N_list),len(number_element_list)])

for idx_ni,ni in enumerate(N_list):

    x_nodes = legendre.nodes(ni)
    

    for idx_e,e in enumerate(number_element_list):

        x_total = ad_mat.total_grid_points(e,x_nodes,x_left,x_right)

        h = (x_total[-1]-x_total[0])/e

        Mk_inv,S = set_up_A_mat(x_nodes,h)
        A = ad_mat.A_mat(Mk_inv,S,a,alpha,e)
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

plt.show()

