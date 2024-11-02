import numpy as np
from numpy.linalg import inv,solve
import matplotlib.pyplot as plt

r1 = 1
r2 = 3

def g_exact(w,theta):
    r = (r2-r1)*(w+1)/2+r1
    return (r+r1**2/r)*np.cos(theta)

import sys
sys.path.insert(0,r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\2_semester\Advanced nummerical\AdvSciComp\Assignment2\func")
sys.path.append(r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\2_semester\Advanced nummerical\AdvSciComp\Assignment2")
import legendre as l
import fourier as f


def solve_PDE(Nw,Ntheta):
    w = l.nodes(Nw)          # r expressed in w
    theta = f.nodes(Ntheta)  # theta
    
    # r variable
    Vander_w,Vw,_ = l.vander(w, Normalize=False)
    D_w     = Vw @ inv(Vander_w)    # Computing differentiation matrix
    D_theta = f.diff_matrix(Ntheta) # Computing differentiation matrix directly
    
    
    # Setting up grid based on the 2 variables
    X,Y = np.meshgrid(w,theta)
    
    # Tensor Product
    Dx = np.kron(np.eye(Ntheta),D_w)
    Dy = np.kron(D_theta, np.eye(Nw))
    
    # Compute the right-hand side
    b = 0*g_exact(X,Y)
    
    # Identify boundary indices (first and last rows/columns in 2D)
    bc_idx_x,bc_idx_y = np.where((X == w[0]) | (X == w[-1]))
    
    # Set boundary condition 
    b[bc_idx_x,bc_idx_y] = g_exact(X[bc_idx_x,bc_idx_y],Y[bc_idx_x,bc_idx_y])
    
    # Right hand side
    b = b.ravel()
    
    # Laplacian Operator with boundary condition
    r = ((r2-r1)*(w+1)/2 + r1)
    c = 2/(r2-r1)
    R_inv = np.kron(np.eye(Ntheta),np.diag(1/r))
    A = c*R_inv@Dx+c**2*Dx@Dx+R_inv@R_inv@Dy@Dy
    
    #term1 = c*np.kron(np.eye(Ntheta),np.diag(1/r)@D_w)
    #term2 = c**2*np.kron(np.eye(Ntheta),D_w@D_w)
    #term3 = np.kron(D_theta@D_theta,np.diag(1/r)@np.diag(1/r)@np.eye(Nw))
    #A =  term1 + term2 + term3
    
    
    # Boundary conditions
    # Compute single index
    bc_idx = bc_idx_x * Nw + bc_idx_y # This should be correct now (compare with b1 = b.ravel())
    
    # Inserting BC     
    A[bc_idx] = 0.0 # All set to 0
    A[bc_idx,bc_idx] = 1.0  # Change the specific index to 1
    
    # Solve linear system
    U = solve(A,b)
    U = U.reshape(Ntheta,Nw)
    return X,Y,U


#%%
"""
Plot of solution
"""

Nw = 32
Ntheta = 32

X,Y,U = solve_PDE(Nw,Ntheta)


# Exact solution
U_exact = g_exact(X, Y)

# Error
error = U - U_exact

# Create a plot with 3 columns and 1 row
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Numerical solution plot
im1 = axes[0].pcolormesh(X, Y, U, shading='auto', cmap='viridis')
axes[0].set_title("Numerical Solution")
axes[0].set_xlabel(r"$w$: space")
axes[0].set_ylabel(r"$\theta$: space")
fig.colorbar(im1, ax=axes[0])

# Exact solution plot
im2 = axes[1].pcolormesh(X, Y, U_exact, shading='auto', cmap='viridis')
axes[1].set_title("Exact Solution")
axes[1].set_xlabel(r"$w$: space")
axes[1].set_ylabel(r"$\theta$: space")
fig.colorbar(im2, ax=axes[1])

# Error plot
im3 = axes[2].pcolormesh(X, Y, error, shading='auto', cmap='viridis')
axes[2].set_title("Error")
axes[2].set_xlabel(r"$w$: space")
axes[2].set_ylabel(r"$\theta$: space")
fig.colorbar(im3, ax=axes[2])

# Display the plot
plt.tight_layout()

#%%

"""
Plot of solution in polar coordinates
"""

R = (r2-r1)*(X+1)/2+r1

# Font size variables
title_fontsize = 18
tick_fontsize = 20
colorbar_tick_fontsize = 18
radial_tick_fontsize = 18


angular_ticks = [np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 
                 5 * np.pi / 4, 3 * np.pi / 2, 7 * np.pi / 4]
angular_labels = [r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', 
                  r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{3\pi}{2}$', r'$\frac{7\pi}{4}$']


fig, axes = plt.subplots(1, 3, figsize=(14, 4), subplot_kw={'projection': 'polar'})


radial_ticks = [1.0, 2.0, 3.0]

# Numerical solution plot
im1 = axes[0].pcolormesh(Y, R, U, shading='auto', cmap='viridis')
axes[0].set_title("Numerical Solution", va='bottom', fontsize=title_fontsize)
axes[0].set_ylim(0, r2)
axes[0].set_xticks(angular_ticks)
axes[0].set_xticklabels(angular_labels, fontsize=tick_fontsize)
axes[0].set_yticks(radial_ticks)
axes[0].tick_params(axis='y', labelsize=radial_tick_fontsize)
cbar1 = fig.colorbar(im1, ax=axes[0])
cbar1.ax.tick_params(labelsize=colorbar_tick_fontsize)

# Exact solution plot
im2 = axes[1].pcolormesh(Y, R, U_exact, shading='auto', cmap='viridis')
axes[1].set_title("Exact Solution", va='bottom', fontsize=title_fontsize)
axes[1].set_ylim(0, r2)
axes[1].set_xticks(angular_ticks)
axes[1].set_xticklabels(angular_labels, fontsize=tick_fontsize)
axes[1].set_yticks(radial_ticks)  # Set fewer radial ticks
axes[1].tick_params(axis='y', labelsize=radial_tick_fontsize)
cbar2 = fig.colorbar(im2, ax=axes[1])
cbar2.ax.tick_params(labelsize=colorbar_tick_fontsize)

# Error plot
im3 = axes[2].pcolormesh(Y, R, error, shading='auto', cmap='viridis')
axes[2].set_title("Error", va='bottom', fontsize=title_fontsize)
axes[2].set_ylim(0, r2)
axes[2].set_xticks(angular_ticks)
axes[2].set_xticklabels(angular_labels, fontsize=tick_fontsize)
axes[2].set_yticks(radial_ticks)  # Set fewer radial ticks
axes[2].tick_params(axis='y', labelsize=radial_tick_fontsize)
cbar3 = fig.colorbar(im3, ax=axes[2])
cbar3.ax.tick_params(labelsize=colorbar_tick_fontsize)

# Display the plot
plt.tight_layout()
plt.show()


#%% Convergence test

#from L2space import discrete_L2_norm

N_list = np.arange(4,32,2)
error = np.zeros_like(N_list,dtype=float)
for idx,N in enumerate(N_list):
    
    # Solve PDE
    X,Y,U = solve_PDE(N,N)

    # Exact solution
    U_exact = g_exact(X, Y)

    # Error
    error[idx] = np.max(np.abs(U-U_exact))

# Define variables to control font size, tick size, and point size
font_size = 12        # Font size for title, labels, legend
tick_size = 12        # Font size for tick labels
point_size = 10       # Size of the points in the line plot

# Plot
plt.figure(figsize=(6, 4))  # Keep the figure size small
plt.title("Convergence Plot", fontsize=font_size + 2)
plt.semilogy(N_list, error, ".-", label=r"$\max|U - U_{exact}|$", markersize=point_size)
plt.xlabel(r"$N$", fontsize=font_size)
plt.ylabel(r"Error", fontsize=font_size)
plt.legend(fontsize=font_size)

# Set tick parameters
plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)

# Adjust margins to add more white space around the plot
plt.subplots_adjust(left=0.15, right=0.9, top=0.85, bottom=0.15)

plt.show()




