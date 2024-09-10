import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from scipy.special import factorial as fac

def convergence_list(N_list,uk_func):

    trunc_err = []

    for N in N_list:

        k_lin_pos = np.arange(N/2+1,10001)
        k_lin_neg = np.arange(-10000,-(N/2+1)+1)

        trunc_err.append(2*np.pi*sum(np.abs(uk_func(np.concatenate([k_lin_neg,k_lin_pos])))**2))

    return trunc_err

def fourier_approx(k_lin,x_lin,uk):

    u_approx_func = lambda x: np.sum(uk*np.exp(1j*k_lin*x))
    u_approx = np.array([u_approx_func(x) for x in x_lin])

    return u_approx

def discrete_fourier_coefficients(u_func,N):

    k_lin = np.arange(-N/2,N/2)
    uk_approx   = np.zeros_like(k_lin,dtype=complex)
    h           = 2*np.pi/(N+1)

    for k_idx,k in enumerate(k_lin):
        s = 0
        for j in range(N):
            xj = j*h
            s += u_func(xj)*np.exp((-2*np.pi*1j*j*k)/N)

        uk_approx[k_idx] = s/N

    return uk_approx

def match_uk(uk,uk_approx):

    return np.allclose(uk,uk_approx,rtol=0.1,atol=0.1)


if __name__ == "__main__":

    # Analytical function 
    u_func = lambda x: 1/(2-np.cos(x)) # Remove pi, to let x = [0:2pi]
    #u_func = lambda x: x
    
    # Fourier coefficients
    uk_func = lambda k: 1/np.sqrt(3)*(2-np.sqrt(3))**np.abs(k)
    
    #uk_func = lambda k: np.where(k == 0, np.pi + 0j, 1j / k) # test kode eksempel fra slide

    # Initializations
    N = 64
    N_list = np.arange(2,N,2) # Only even numbers
    trunc_err = convergence_list(N_list,uk_func)
    k_lin = np.arange(-N/2,N/2)
    
    # Convergence plot
    plt.figure(1)
    plt.semilogy(N_list,trunc_err)
    plt.xlabel("N")
    plt.ylabel(r"$||\tau||^2$")
    plt.title("Convergence plot")

    # Discrete fourier transform

    uk_approx = {} # Using dictionary
    for Ni in np.array([4,8,16,32,64,128]): # Changing the size of N that is the number of waves
        uk_approx[Ni] = discrete_fourier_coefficients(u_func,Ni) # Discrete fourier transformation
        k_lin_temp = np.arange(-Ni/2,Ni/2) 
        match = match_uk(uk_func(k_lin_temp),uk_approx[Ni])      # Matching the discrete FT with the analytical
        print(match)

    # Plotting the analytical uk vs the approximated
    plt.figure(2)
    plt.plot(k_lin_temp,uk_approx[Ni],"o-",label=f"N={Ni}")
    plt.plot(k_lin_temp,uk_func(k_lin_temp),"o-",label="uk analytical")
    plt.xlabel("k")
    plt.ylabel("uk")
    plt.legend()

    x_lin = np.linspace(0,2*np.pi,Ni)

    # Approximating function using fourier coefficients
    u_approx_analytical = fourier_approx(k_lin_temp,x_lin,uk_func(k_lin_temp))
    u_approx_discrete = fourier_approx(k_lin_temp,x_lin,uk_approx[Ni])

    # Plotting the approximated function values
    plt.figure(3)
    plt.plot(x_lin,u_approx_analytical.real,label="u analytical")
    plt.plot(x_lin,u_approx_discrete.real,label="u discrete")
    plt.legend()
    plt.show()



