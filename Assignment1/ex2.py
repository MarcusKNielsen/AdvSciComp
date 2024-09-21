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

        ax1.plot(x,gamma(n+1)*gamma(0.5)/gamma(n+0.5) * cheby,label=rf"$T_{{{n}}}^{{(-1/2,-1/2)}}$")
        ax2.plot(x,legendre,label=rf"$P_{{{n}}}^{{(0,0)}}$")

        ax1.set_title("Chebyshevs polynomials")
        ax2.set_title("Legendre polynomials")

        ax1.set_xlabel("x")
        ax2.set_xlabel("x")

        ax1.legend()
        ax2.legend()

        plt.tight_layout()

def uk_approx_func(u_func,k_list,xj,N,alpha=0,beta=0):
    
    K = len(k_list)
    uk_approx = np.zeros(K)

    wj = (1-xj)**alpha*(1+xj)**beta
    for k_idx, k in enumerate(k_list): 
        # uk_temp = 0
        #yk = 0
        phi_k = JacobiP(xj,alpha=0,beta=0,N=k)
        uk_approx[k_idx] = np.sum((u_func(xj))*phi_k*wj) / np.sum(phi_k * phi_k * wj) 
        
        # for j in range(N):
        #     wj = (1-xj[j])**alpha*(1+xj[j])**beta
        #     phi_k = JacobiP(xj[j],alpha=0,beta=0,N=k)
        #     uk_temp += (u_func(xj[j]))*phi_k*wj
        #     yk += phi_k**2*wj 

        #uk_approx[k_idx] = uk_temp /yk
    
    return uk_approx

if __name__ == "__main__":

    Jacobi_visualize()


    #%% i 
    # List of N used in convergence plot
    N_list = np.array([10,40,80,100,200])
    # Analytical function 
    u_func = lambda x: 1/(2-np.cos(np.pi*(x+1))) # Remove pi, to let x = [-1:1]
    # Discrete poly coefficients
    #uk_approx_func = lambda k,N: discrete_poly_coefficients(k,N,u_func=u_func)

    #x_lin = np.linspace(-1,1,200)
    k_list = np.arange(0,200)

    uk_approx = np.zeros([len(k_list),len(N_list)])

    fig, axs = plt.subplots(len(N_list), 1, figsize=(7, len(N_list)*2), constrained_layout=True)

    for N_idx, N in enumerate(N_list):
        xj = JacobiGL(alpha=0, beta=0, N=N-1)
        uk_approx[:, N_idx] = uk_approx_func(u_func, k_list, xj, N, alpha=0, beta=0)
        
        axs[N_idx].vlines(N_list[N_idx], min(uk_approx[:, N_idx]), max(uk_approx[:, N_idx]), 
                          linestyle="--", color="red", label=f"N = {N_list[N_idx]}")
        axs[N_idx].plot(k_list, uk_approx[:, N_idx], ".-", label=r"$u_k$")
        
        
        axs[N_idx].set_title(f"Plot for N = {N}")
        axs[N_idx].legend()
    
    # Set a shared x-label for all subplots
    plt.xlabel('k_list')
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




