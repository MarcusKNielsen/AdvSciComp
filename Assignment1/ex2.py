import numpy as np 
import matplotlib.pyplot as plt


from ex1 import check_N,convergence_list,fourier_approx
from JacobiGL import JacobiGL

def w0(x):
    x = x/(2*np.pi)
    return (x < 0)*(-np.cos(x)) + (x >= 0)*np.cos(x)

def w1(x):
    x = x/(2*np.pi)
    return np.sin(np.maximum(x,0)) - np.sin(np.minimum(x,0))

def w2(x):
    x = x/(2*np.pi)
    return -np.cos(np.maximum(x,0)) + np.cos(np.minimum(x,0))

def w3(x):
    x = x/(2*np.pi)
    return - np.sin(np.abs(x)) + np.abs(x) - 2*np.pi

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

def GradJacobiP(x,alpha,beta,N):

    if N == 0:
        return 0

    return 1/2*(alpha+beta+N+1)*JacobiP(x,alpha+1,beta+1,N-1)

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

def discrete_inner_product(u,v,w):    
    return np.sum(u*v*w)
    

def discrete_L2_norm(u,w):
    return np.sqrt(discrete_inner_product(u,u,w))


"""
Nedenstående funktioner virker kun for legendre (alpha=0,beta=0).
"""

def uk_approx_func(u_func,k_list,xj,N,alpha=0,beta=0):
    # This function works only for legendre because of the weights
    K = len(k_list)
    uk_approx = np.zeros(K)

    wj = 2/((N-1)*N) * 1/JacobiP(xj,alpha=0,beta=0,N=N)**2
    for k_idx, k in enumerate(k_list): 
        phi_k = JacobiP(xj,alpha=0,beta=0,N=k)
        uk_approx[k_idx] = np.sum(u_func(xj)*phi_k*wj) / np.sum(phi_k * phi_k * wj) 
    
    return uk_approx

def poly_approx(k_lin,x_GL,uk):
    u_approx = np.zeros_like(x_GL)
    for k in k_list:
        u_approx += uk[k] * JacobiP(x_GL,alpha=0,beta=0,N=k)
    return u_approx

def get_vandermonde(x_GL):
    N = len(x_GL)
    V = np.zeros([N,N])
    for j in range(N):
        V[:,j] = JacobiP(x_GL,alpha=0,beta=0,N=j)
    return V

def get_vandermonde_x(x_GL):
    N = len(x_GL)
    V = np.zeros([N,N])
    for j in range(N):
        V[:,j] = GradJacobiP(x_GL,alpha=0,beta=0,N=j)
    return V

def D_poly(x_GL):
    Vx = get_vandermonde_x(x_GL)
    V = get_vandermonde(x_GL)

    return Vx@np.linalg.inv(V)

def get_extended_vandermonde(x,N):
    K = len(x)
    Vm = np.zeros([K,N])
    for j in range(N):
        Vm[:,j] = JacobiP(x,alpha=0,beta=0,N=j)
    return Vm

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

        xj = JacobiGL(alpha=0, beta=0, N=N)
        uk_approx[:, N_idx] = uk_approx_func(u_func, k_list, xj, N, alpha=0, beta=0)
        
        axs[N_idx].vlines(N_list[N_idx], min(uk_approx[:, N_idx]), max(uk_approx[:, N_idx]), 
                          linestyle="--", color="red", label=f"N = {N_list[N_idx]}")
        axs[N_idx].plot(k_list, uk_approx[:, N_idx], ".-", label=r"$u_k$")
        
        
        axs[N_idx].set_title(f"Plot for N = {N}")
        axs[N_idx].legend()
    
    # Set a shared x-label for all subplots
    plt.xlabel('k_list')
    #plt.show()
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

    ##plt.show(()

    #%% j approx sin(pi x)
    
    v_func = lambda x: np.sin(np.pi * x)
    N_points = 40
    
    N = N_points - 1
    k_list = np.arange(0,N)
    x_GL = JacobiGL(alpha=0, beta=0, N=N)
    vk_approx = uk_approx_func(v_func,k_list,x_GL,N,alpha=0,beta=0)
    
    u_approx = poly_approx(k_list,x_GL,vk_approx)
    
    
    plt.figure()
    x = np.linspace(-1,1,1000)
    plt.plot(x_GL,u_approx,".-",label="Approx")
    plt.plot(x,v_func(x),label=r"$\sin(\pi x)$")
    plt.xlabel("x")
    plt.legend()
    #plt.show(()
    
    # Convergence plot
    N_list = np.arange(2,50,2)
    err = np.zeros_like(N_list,dtype=float)
    for N_idx,N in enumerate(N_list):
        
        # Compute approximation
        k_list = np.arange(0,N)
        x_GL = JacobiGL(alpha=0, beta=0, N=N)
        vk_approx = uk_approx_func(v_func,k_list,x_GL,N,alpha=0,beta=0)
        u_approx = poly_approx(k_list,x_GL,vk_approx)
        
        # Setup Vandermonde matrices
        V = get_vandermonde(x_GL)
        V_inv = np.linalg.inv(V)
        x = np.linspace(-1,1,100)
        Vm = get_extended_vandermonde(x,N+1)
        
        # Compute true values
        u_true = v_func(x)
        u_true_on_GL = v_func(x_GL)
        
        
        # compute discrete L2 - error 
        err[N_idx] = np.max(np.abs(u_true-Vm@V_inv@u_approx))
        #err[N_idx] = np.max(np.abs(u_true_on_GL-u_approx))
    
    plt.figure()
    plt.semilogy(N_list,err,".-",label=r"$\max_x | \sin(\pi x) - \tilde{V} \ V^{-1} \bar{u}|$")
    plt.xlabel("N")
    plt.ylabel("error")
    plt.legend()
    #plt.show(()
    
    #%% j Extrapolation
    
    v_func = lambda x: np.sin(np.pi * x)
    N_points = 40
    
    N = N_points - 1
    k_list = np.arange(0,N)
    x_GL = JacobiGL(alpha=0, beta=0, N=N)
    vk_approx = uk_approx_func(v_func,k_list,x_GL,N,alpha=0,beta=0)
    
    u_approx = poly_approx(k_list,x_GL,vk_approx)
    
    # Setup Vandermonde matrices on larger domain
    V = get_vandermonde(x_GL)
    V_inv = np.linalg.inv(V)
    x = np.linspace(-1.4,1.4,100)
    Vm = get_extended_vandermonde(x,N+1)
    
    plt.figure()
    plt.plot(x,Vm@V_inv@u_approx,".-",label=r"$\tilde{V} \ V^{-1} \bar{u}$")
    plt.plot(x,v_func(x),label=r"$\sin(\pi x)$")
    plt.xlabel("x")
    plt.legend()
    plt.title("Extrapolation")
    ##plt.show(()

    norm_error = []
    for Ni in np.array([10,20,60,100,200]):

        x_GL = JacobiGL(alpha=0,beta=0,N=Ni)
        # w'(x)-Dw(x)
        expression = w2(x_GL)-D_poly(x_GL)*w3(x_GL)
        weights    = 2/((Ni-1)*Ni)* 1/(JacobiP(x_GL,alpha=0,beta=0,N=Ni)**2)
        norm_error.append(discrete_L2_norm(expression,weights))
    
    
    plt.figure()
    plt.semilogy(np.array([10,20,60,100,200]), norm_error)
    plt.xlabel("N")
    plt.ylabel("$||w'-Dw||_N^2$")
    plt.show()





