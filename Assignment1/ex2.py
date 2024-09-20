import numpy as np 
import matplotlib.pyplot as plt


from ex1 import check_N,convergence_list,fourier_approx
from JacobiGL import JacobiGL

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


def Jacobi_visualize(N=6):
    from scipy.special import gamma
    x = np.linspace(-1,1,1000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    for n in range(N):

        cheby = JacobiP(x,alpha=-1/2,beta=-1/2,N=n)
        legendre = JacobiP(x,alpha=0,beta=0,N=n)

        ax1.plot(x,gamma(n+1)*gamma(0.5)/gamma(n+0.5) * cheby,label=r"$P_{n}^{(-1/2,-1/2)}$")
        ax2.plot(x,legendre,label=r"$P_{n}^{(0,0)}$")

        ax1.set_title("Chebyshevs polynomials")
        ax2.set_title("Legendre polynomials")

        ax1.set_xlabel("x")
        ax2.set_xlabel("x")

        ax1.legend()
        ax2.legend()

        plt.tight_layout()

def uk_approx_func(x,xj,N,alpha=0,beta=0):

    uk_approx = np.zeros([len(x)])

    for k in np.arange(len(x)): 
        uk_temp = 0
        yk = 0
        for j in range(N):
            wj = (1-xj[j])**alpha+(1+xj[j])**beta
            phi_k = JacobiP(xj[j],alpha=0,beta=0,N=j)
            uk_temp += (u_func(xj[j]))*phi_k*wj
            yk += phi_k**2*wj 

        uk_approx[k] = uk_temp /yk
    
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
    #uk_approx_func = lambda k,N: discrete_poly_coefficients(k,N,u_func=u_func)

    x_lin = np.linspace(-1,1,200)

    uk_approx = np.zeros([len(x_lin),len(N_list)])

    plt.figure(2)

    for N_idx,N in enumerate(N_list):

        xj = JacobiGL(alpha=0, beta=0, N=N-1)
        uk_approx[:,N_idx] = uk_approx_func(x_lin,xj,N,alpha=0,beta=0)

        plt.plot(x_lin,uk_approx[:,N_idx],label="N")

    plt.legend()
    plt.show()

    #trunc_err = convergence_list_poly(N_list,poly_approx,u_func,uk_approx_func)
    #plt.figure(2)
    #plt.semilogy(N_list,trunc_err,"o-",label=r"Numerical: $||u - I_Nu ||^2$")
    #plt.xlabel("N")
    #plt.ylabel(r"$||\tau||^2$")
    #plt.legend(fontsize=12) 
    #plt.title("Convergence Plot (logarithmic y-axis)")

    #%% j 

    # Construction of Vandermonde matrix:

    N = 7 # given grid points

    x_GL = JacobiGL(alpha=0, beta=0, N=6) # Grid points

    V = np.zeros([N,N])
    for j in range(N):
        V[:,j] = JacobiP(x_GL,alpha=0,beta=0,N=j)

    # Larger Vandemonde matrix
    Vm = np.zeros([100,N])
    xV_lin = np.linspace(-1,1,100)
    for j in range(N):
        Vm[:,j] = JacobiP(xV_lin,alpha=0,beta=0,N=j)

    # Visualising lagrange polynomials
    V_inv = np.linalg.inv(V)
    x_lin = np.linspace(-1,1,100)

    plt.figure(3,figsize=(8,4))
    for n in range(N):
        
        f_streg = np.zeros(N)
        f_streg[n] = 1

        f_hat = Vm@V_inv@f_streg

        plt.plot(x_lin,f_hat,label=f"$h_{n}$")
    
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("x")
    plt.title("Lagrange Polynomials Obtained by the Vandermonde Matrix")

    plt.show()




