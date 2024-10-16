import numpy as np
from numpy.linalg import inv,solve
import matplotlib.pyplot as plt

r1 = 1
r2 = 3

def g_exact(w,theta):
    r = 2*(w-r1)/(r2-r1)-1
    return (r+r1**2/r)*np.cos(theta)

#import sys
#sys.path.insert(0,r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\2_semester\Advanced nummerical\AdvSciComp\Assignment2\func")
#sys.path.append(r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\2_semester\Advanced nummerical\AdvSciComp\Assignment2")
import legendre as l
import fourier as f


"""
The error seems to large
"""

Nw = 10
Ntheta = 15
w = l.nodes(Nw)          # r expressed in w
theta = f.nodes(Ntheta)  # theta

# r variable
Vander_w,Vw,_ = l.vander(w)
D_w = Vw @ inv(Vander_w)    # Computing differentiation matrix
D_theta = f.diff_matrix(Ntheta) # Computing differentiation matrix directly

# Setting up grid based on the 2 variables
X,Y = np.meshgrid(w,theta)

# Tensor Product
Dx = np.kron(D_w, np.eye(Ntheta))
Dy = np.kron(np.eye(Nw), D_theta)

# Compute the right-hand side
b = 0*g_exact(X,Y)

# Identify boundary indices (first and last rows/columns in 2D)
bc_idx_x,bc_idx_y = np.where((X == w[0]) | (X == w[-1]))

# Set boundary condition 
b[bc_idx_x,bc_idx_y] = g_exact(X[bc_idx_x,bc_idx_y],Y[bc_idx_x,bc_idx_y])

# Right hand side
b = b.ravel()

# Laplacian Operator with boundary condition
r = (2*(X-r1)/(r2-r1)-1).ravel()
multi1 = 1/r*(r2-r1)/2*Dx.T
multi2 = (1/r)**2*(Dy@Dy).T
A = multi1.T+Dx@Dx+multi2.T

#%% Boundary conditions

# Compute single index
bc_idx = bc_idx_y * Ntheta + bc_idx_x # OBS NTheta eller NW!!!!

# Inserting BC     
A[bc_idx] = 0
A[bc_idx,bc_idx] = 1

# Solve linear system
U = solve(A,b)
U = U.reshape(Ntheta,Nw)

# Exact solution
U_exact = g_exact(X, Y)

# Error
error = U - U_exact

# Create a plot with 3 columns and 1 row
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Numerical solution plot
im1 = axes[0].pcolormesh(X, Y, U, shading='auto', cmap='viridis')
axes[0].set_title("Numerical Solution")
axes[0].set_xlabel("x: space")
axes[0].set_ylabel("y: space")
fig.colorbar(im1, ax=axes[0])

# Exact solution plot
im2 = axes[1].pcolormesh(X, Y, U_exact, shading='auto', cmap='viridis')
axes[1].set_title("Exact Solution")
axes[1].set_xlabel("x: space")
axes[1].set_ylabel("y: space")
fig.colorbar(im2, ax=axes[1])

# Error plot
im3 = axes[2].pcolormesh(X, Y, error, shading='auto', cmap='viridis')
axes[2].set_title("Error")
axes[2].set_xlabel("x: space")
axes[2].set_ylabel("y: space")
fig.colorbar(im3, ax=axes[2])

# Display the plot
plt.tight_layout()
plt.show()


