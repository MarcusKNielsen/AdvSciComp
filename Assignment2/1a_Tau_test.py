import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,r"/home/max/Documents/DTU/AdvNumericalMethods/AdvSciComp/Assignment2/func")
sys.path.insert(0,r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\2_semester\Advanced nummerical\AdvSciComp\Assignment2\func")
sys.path.append(r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\2_semester\Advanced nummerical\AdvSciComp\Assignment2")
import legendre
import L2space

def a_coefs(N,eps):

    n = np.arange(N)
    result = np.zeros_like(n, dtype=float)

    # p = 0
    result[n == 0] = 0

    # p even
    even_condition = (n % 2 == 0) & (n != 0)
    n_even = n[even_condition] # indices to multiply
    result[even_condition] = -2 * eps * n_even * (n_even + 1)

    # p odd
    odd_condition = n % 2 != 0
    result[odd_condition] = -2
    
    return result

def b_coefs(N,eps):

    n = np.arange(N)
    result = np.zeros_like(n, dtype=float)

    # p = 0,1
    result[n == 0] = 0
    result[n == 1] = 0

    # p even
    even_condition = (n % 2 == 0) & (n != 0)
    n_even = n[even_condition] # indices to multiply
    result[even_condition] = eps * (n_even * (n_even + 1) - 2)

    # p odd
    odd_condition = (n % 2 != 0) & (n != 1)
    result[odd_condition] = 1
    
    return result

# parameters
N = 32
eps = 0.1

x = legendre.nodes(N)
V,Vx,w = legendre.vander(x,N)

# Setup A matrix
A = np.zeros([N,N])


# The n=0 equation
A[0] = a_coefs(N,eps)

# The n=1 equation
A[1] = b_coefs(N,eps)

# three term part
for n in range(2,N-2):
    A[n,n-1] = 1/(2*n-1)
    A[n,n]   = 2*eps
    A[n,n+1] = -1/(2*n+3)

# boundary condition
#A[-2] = V[0]
#A[-1] = V[-1]

A[-2] = np.ones_like(N)
A[-1] = np.ones_like(N)
n = np.arange(N)
odd_condition = (n % 2 != 0)
A[-1,odd_condition] = -1

# Compute hat_f_0
hat_f_0 = L2space.discrete_inner_product(np.ones(N),V[:,0],w)

# Setup right hand side
b = np.zeros(N)
#b[0] =  hat_f_0
#b[2] = -hat_f_0/6

b[0] =  1
b[2] = -1/6


# solve for coefficients
u_hat = solve(A,b)

# plot solution
plt.plot(x,V@u_hat)
plt.show()




