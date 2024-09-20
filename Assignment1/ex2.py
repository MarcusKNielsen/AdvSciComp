import numpy as np 
import matplotlib.pyplot as plt

from ex1 import check_N,convergence_list,fourier_approx
from JacobiGL import JacobiGL

def JacobiP(x,alpha,beta,N):

    J_nm2 = np.ones(len(x)) # J_{n-2}
    J_nm1 = 1/2*(alpha-beta+(alpha+beta+2)*x) # J_{n-1}

    if N==0:
        return J_nm2
    elif N==1:
        return J_nm1

    for n in range(2,N+1):

        # Computing the recursive coefficients
        anm2 = 2*(n+alpha)*(n+beta)/( (2*n+alpha+beta+1)*(2*n+alpha+beta) )
        anm1  = (alpha**2-beta**2)/( (2*n+alpha+beta+2)*(2*n+alpha+beta) )
        an = 2*(n+1)*(n+beta+alpha+1)/( (2*n+alpha+beta+2)*(2*n+alpha+beta+1) )

        # Computing
        J_n = ( (anm1 + x )*J_nm1 - anm2*J_nm2 ) / an

        # Updating step
        J_nm2 = J_nm1
        J_nm1 = J_n
    
    return J_n

def Jacobi_visualize(N=6):

    x = np.linspace(-1,1,1000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    for n in range(N):

        cheby = JacobiP(x,alpha=-1/2,beta=-1/2,N=n)
        legendre = JacobiP(x,alpha=0,beta=0,N=n)

        ax1.plot(x,cheby,label=f"$P_{n}^{(-1/2,-1/2)}$")
        ax2.plot(x,legendre,label=f"$P_{n}^{(0,0)}$")

        ax1.set_title("Chebyshevs polynomials")
        ax2.set_title("Legendre polynomials")

        ax1.set_xlabel("x")
        ax2.set_ylabel("x")

        ax1.legend()
        ax2.legend()

        plt.tight_layout()

def discrete_poly_coefficients(k_lin,alpha=0,beta=0,u_func=lambda x: 1/(2-np.cos(x))):

    N = len(k_lin) 
    uk_approx   = np.zeros_like(k_lin,dtype=complex)
        
    for k_idx,k in enumerate(k_lin): 
        s = 0
        yk = 0
        for j in range(N):
            xj = 2*np.pi*j/N 
            wj = (1-xj)**alpha+(1+xj)**beta
            phi_k = JacobiP(xj,alpha=0,beta=0,N=j)
            s += (u_func(xj))*phi_k*wj
            yk += phi_k**2*wj 

        uk_approx[k_idx] = s/yk

    return uk_approx


if __name__ == "__main__":

    #%% h : (OBS: Der er lidt galt, fordi de matcher ikke helt med dem på slides, tror der er noget scale som er problemet)
    Jacobi_visualize()


    #%% i 
    # List of N used in convergence plot
    N_list = np.array([10,40,80,100,200])
    # Analytical function 
    u_func = lambda x: 1/(2-np.cos(x)) # Remove pi, to let x = [0:2pi]
    # Fourier coefficients
    uk_func = lambda k: 1/(np.sqrt(3)*(2+np.sqrt(3))**np.abs(k))
    # Discrete poly coefficients
    uk_approx_func = lambda k: discrete_poly_coefficients(k,u_func=u_func)


    trunc_err = convergence_list(N_list,fourier_approx,u_func,uk_func)

    plt.figure(2)
    plt.semilogy(N_list,trunc_err,"o-",label=r"Numerical: $||u - I_Nu ||^2$")
    plt.xlabel("N")
    plt.ylabel(r"$||\tau||^2$")
    plt.legend(fontsize=12) 
    plt.title("Convergence Plot (logarithmic y-axis)")

    #%% j 

    # Construction of Vandermonde matrix:

    N = 7 # given grid points

    x_GL = JacobiGL(alpha=0, beta=0, N=6) # Grid points

    V = np.zeros([N,N])
    for j in range(N):
        V[j,:] = JacobiP(x_GL,alpha=0,beta=0,N=j)

    # Larger Vandemonde matrix
    Vm = np.zeros([100,N])
    for j in range(100):
        Vm[j,:] = JacobiP(x_GL,alpha=0,beta=0,N=j)

    # Visualising lagrange polynomials
    V_inv = np.linalg.inv(V)
    x_lin = np.linspace(-1,1,100)

    plt.figure(3)
    for n in range(N):
        fs = np.zeros(N)
        fs[n] = 1

        fm = Vm@V_inv@fs

        plt.plot(x_lin,fm*JacobiP(x_lin,0,0,n),label=f"$h_{n}$")
        
    plt.legend()    


    plt.show()




