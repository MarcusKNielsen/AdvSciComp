import numpy as np
from numpy.linalg import solve
import fourier
import matplotlib.pyplot as plt


N = 65
x = fourier.nodes(N)

D = fourier.diff_matrix(N)

A = D@D
A[0] = 0
A[0,0] = 1

b = -np.sin(x)
b[0] = 0

u = solve(A,b)

plt.figure()
x_exact = np.linspace(0,2*np.pi,200)
plt.plot(x,u,".",label="approx")
plt.plot(x_exact,np.sin(x_exact),label="exact")
plt.legend()
plt.xlabel("x")
plt.title(r"Poisson Problem: $\nabla^2u(x) = f(x)$")
plt.show()

print(f"error = {np.max(np.abs(u - np.sin(x)))}")









