import numpy as np
from numpy.linalg import inv,solve
import matplotlib.pyplot as plt

r1 = 1
r2 = 2

def u_exact(x,y):
    r = (r2-r1)*(x+1)/2+r1
    theta = np.pi*(y+1)
    return (r+r1**2/r)*np.cos(theta)

#import sys
#sys.path.insert(0,r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\2_semester\Advanced nummerical\AdvSciComp\Assignment2\func")
#sys.path.append(r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\2_semester\Advanced nummerical\AdvSciComp\Assignment2")
from legendre import vander, nodes

N = 32
x = nodes(N)
V,Vx,_ = vander(x)
D = Vx @ inv(V)

X,Y = np.meshgrid(x,x)

# Tensor Product
I = np.eye(N)
Dx = np.kron(I,D)
Dy = np.kron(D,I)

# Compute the right-hand side
b = np.zeros([N,N])

# Identify boundary indices (first and last rows/columns in 2D)
bc_idx_x,bc_idx_y = np.where((X == x[0]) | (X == x[-1]) | (Y == x[0]) | (Y == x[-1]))

# Get single index
bc_idx = bc_idx_x * N + bc_idx_y 

# Set boundary condition 
b[bc_idx_x,bc_idx_y] = u_exact(X[bc_idx_x,bc_idx_y],Y[bc_idx_x,bc_idx_y])

# Right hand side
b = b.ravel()

# Laplacian Operator with boundary condition
r = ((r2-r1)*(X+1)/2+r1).ravel()
c = 2/(r2-r1)
A = (c/r) * Dx +  c**2 * Dx@Dx + (1/r**2)*(Dy@Dy) 
#A = np.kron(c/r,I) * Dx +  c**2 * Dx@Dx + np.kron(1/r**2,I)*(Dy@Dy)

A[bc_idx] = 0
A[bc_idx,bc_idx] = 1

# Solve linear system
U = solve(A,b)
U = U.reshape(N,N)

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

plt.figure()
plt.plot(x,U[16])
plt.plot(x,U_exact[16])
plt.show()
