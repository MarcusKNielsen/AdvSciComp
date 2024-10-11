import numpy as np
from numpy.linalg import inv,solve
from legendre import vander
import matplotlib.pyplot as plt
from JacobiGL import JacobiGL


N = 32
x = JacobiGL(0,0,N)

V,Vx,_ = vander(x)


Vinv = inv(V)

D = Vx @ Vinv

eps = 0.1

A = -4*eps*D@D-2*D
A[0,:]   = 0
A[0,0]   = 1
A[-1,:]  = 0
A[-1,-1] = 1

b = np.ones(N)
b[0]  = 0
b[-1] = 0

u = solve(A,b)

def u_exact(x,epsilon):
    x = (x+1)/2
    return (np.exp(-x / epsilon) + (x - 1) - np.exp(-1 / epsilon )*x) / (np.exp(-1 / epsilon) - 1)

plt.figure()
plt.plot(x,u,".-",label="approx")
x_large = np.linspace(-1,1,100)
plt.plot(x_large,u_exact(x_large,eps),"--",label="exact")
plt.legend()
plt.xlabel("x")
plt.show()

print(f"error = {np.max(np.abs(u_exact(x,eps) - u))}")











