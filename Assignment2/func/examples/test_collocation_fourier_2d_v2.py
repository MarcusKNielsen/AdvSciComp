import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt

"""
 - Something weird happens for N even.
"""

#import sys
#sys.path.insert(0,r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\2_semester\Advanced nummerical\AdvSciComp\Assignment2\func")
#sys.path.append(r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\2_semester\Advanced nummerical\AdvSciComp\Assignment2")
#from legendre import vander, nodes
import fourier

def u_exact(x,y):
    return np.sin(x)*np.sin(y)

# x direction
Nx = 31
x  = fourier.nodes(Nx)
Dx = fourier.diff_matrix(Nx)

# y direction
Ny = 51
y  = fourier.nodes(Ny)
Dy = fourier.diff_matrix(Ny)

X,Y = np.meshgrid(x,y)

# Tensor Product
Dx = np.kron(np.eye(Ny),Dx)
Dy = np.kron(Dy,np.eye(Nx))

# Compute the right-hand side
b = (-2)*u_exact(X,Y)

# Identify boundary indices (first and last rows/columns in 2D)
bc_idx_x,bc_idx_y = np.where((X == x[0]) | (Y == y[0]))


# Get single index
bc_idx = bc_idx_x * Nx + bc_idx_y

# Set boundary condition
b[bc_idx_x,bc_idx_y] = u_exact(X[bc_idx_x,bc_idx_y],Y[bc_idx_x,bc_idx_y])

# Right hand side
b = b.ravel()

# Laplacian Operator with boundary condition
A = Dx@Dx + Dy@Dy
A[bc_idx] = 0
A[bc_idx,bc_idx] = 1

# Solve linear system
U = solve(A,b)

U = U.reshape(Ny,Nx)

# Exact solution
U_exact = u_exact(X, Y)

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


