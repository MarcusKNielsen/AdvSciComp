import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,r"/home/max/Documents/DTU/AdvNumericalMethods/AdvSciComp/Assignment2/func")
#sys.path.insert(0,r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\2_semester\Advanced nummerical\AdvSciComp\Assignment2\func")
#sys.path.append(r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\2_semester\Advanced nummerical\AdvSciComp\Assignment2")
import legendre

def a_coefs(N,eps):

    n = np.arange(N)
    result = np.zeros_like(n, dtype=float)
    result[n == 0] = 0
    even_condition = (n % 2 == 0) & (n != 0)
    n_even = n[even_condition]
    result[even_condition] = -2 * eps * n_even * (n_even + 1)
    result[n % 2 != 0] = -2
    
    return result

# parameters
N = 50
eps = 0.1

x = legendre.nodes(N)
V,_,_ = legendre.vander(x,N)

# Setup A matrix
A = np.zeros([N,N])


# The n=0 equation
A[0] = a_coefs(N,eps)

# three term part
for n in range(1,N-2):
    A[n,n-1] = 1/(2*n-1)
    A[n,n]   = 2*eps
    A[n,n+1] = -1/(2*n+3)

# boundary condition
A[-2] = V[0]
A[-1] = V[-1]


# Setup right hand side
b = np.zeros(N)
b[0] = 1

# solve for coefficients
u_hat = solve(A,b)

# plot solution
plt.plot(x,V@u_hat)




