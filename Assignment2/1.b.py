import numpy as np
from numpy.linalg import inv,solve
import matplotlib.pyplot as plt

r1 = 1
r2 = 2

def g_exact(w,theta):
    r = (r2-r1)*(w+1)/2+r1
    return (r+r1**2/r)*np.cos(theta)

import sys
sys.path.insert(0,r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\2_semester\Advanced nummerical\AdvSciComp\Assignment2\func")
sys.path.append(r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\2_semester\Advanced nummerical\AdvSciComp\Assignment2")
import legendre as l
import fourier as f

Nw = 50
Ntheta = 50
w = l.nodes(Nw)          # r expressed in w
theta = f.nodes(Ntheta)  # theta

# r variable
Vander_w,Vw,_ = l.vander(w)
D_w     = Vw @ inv(Vander_w)    # Computing differentiation matrix
D_theta = f.diff_matrix(Ntheta) # Computing differentiation matrix directly

# Setting up grid based on the 2 variables
X,Y = np.meshgrid(w,theta)

# Tensor Product
Dx = np.kron(np.eye(Ntheta), D_w)
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
r = ((r2-r1)*(X+1)/2 + r1).ravel()
c = 2/(r2-r1)
A = (2/(r2-r1))*((1/r)*Dx+2/(r2-r1)*Dx@Dx)+1/(r**2)*Dy@Dy #(c*(1/r)) * Dx + c**2 * Dx@Dx + (1/r**2)*(Dy@Dy)

# Alternative
#A1 = (c**2/r) * Dx @ (r*Dx) + (1/r)**2*(Dy@Dy) 

# Another alternative
#I = np.eye(Nw)
#r = (r2-r1)*(w+1)/2 + r1
#A = c * np.kron(I,np.diag(1/r)) * Dx + c**2 * Dx@Dx + np.kron(I,np.diag((1/r)**2))*(Dy@Dy)

#%% Boundary conditions

# Compute single index
bc_idx = bc_idx_x * Nw + bc_idx_y # This should be correct now (compare with b1 = b.ravel())

# Inserting BC     
A[bc_idx] = 0.0 # All set to 0
A[bc_idx,bc_idx] = 1.0  # Change the specific index to 1

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

#%% Convergence test

N_list = np.arange(5,30)
error = []


plt.show()


