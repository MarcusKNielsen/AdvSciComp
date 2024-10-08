import numpy as np
import matplotlib.pyplot as plt
from JacobiGL import JacobiGL

def discrete_inner_product(u,v,w):    
    return np.sum(u*v*w)

def discrete_L2_norm(u,w):
    return np.sqrt(discrete_inner_product(u,u,w))

def JacobiP(x,alpha,beta,N):

    if x.size > 1:
        J_nm2 = np.ones(len(x)) # J_{n-2}
    else:
        J_nm2 = 1
    J_nm1 = 1/2*(alpha-beta+(alpha+beta+2)*x) # J_{n-1}

    if N==0:
        return J_nm2
    elif N==1:
        return J_nm1

    for n in range(1,N):

        # Computing the recursive coefficients
        anm2  = 2*(n+alpha)*(n+beta)/( (2*n+alpha+beta+1)*(2*n+alpha+beta) )
        anm1  = (alpha**2-beta**2)/( (2*n+alpha+beta+2)*(2*n+alpha+beta) )
        an    = 2*(n+1)*(n+beta+alpha+1)/( (2*n+alpha+beta+2)*(2*n+alpha+beta+1) )

        # Computing
        J_n = ( (anm1 + x )*J_nm1 - anm2*J_nm2 ) / an

        # Updating step
        J_nm2 = J_nm1
        J_nm1 = J_n
    
    return J_n

def A_matrix(N,epsi):

    A = np.zeros([N+2,N+2])

    for i in range(1,N+1):
        
        lower_diag  = -2/(2*i-1)
        main_diag = -4*epsi*np.ones_like(i)
        upper_diag = 2/(2*i+1)

        ai = np.zeros(N+2)

        ai[i-1] = lower_diag
        ai[i] = main_diag
        ai[i+1] = upper_diag
    
        A[i] = ai 


    # Inserting boundary conditions:
    # i = 0 and i = N+2
    for j in range(N+2):
        A[0,j] = JacobiP(np.array([-1]),alpha=0,beta=0,N=j)
        A[-1,j] = JacobiP(np.array([1]),alpha=0,beta=0,N=j)

    return A

def f_RHS(N):

    f = np.ones_like(N+2)
    x_GL = JacobiGL(0,0,N+1)
    N_weights = len(x_GL)
    weights = 2/((N_weights-1)*N_weights)* 1/(JacobiP(x_GL,alpha=0,beta=0,N=N_weights-1)**2)
    f_tilde = np.zeros(N+2)
    
    for j in range(1,N+1):

        phi = JacobiP(x_GL,alpha=0,beta=0,N=j)

        f_tilde[j] = discrete_inner_product(f,phi,weights)
    
    return f_tilde


A = A_matrix(3,0.1)
f_tilde = f_RHS(10)

print(A)





