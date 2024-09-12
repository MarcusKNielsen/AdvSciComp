import matplotlib.pyplot as plt
import numpy as np


def convergence_list(N_list,fourier_approx,u_func,uk_func):

    trunc_err = []

    for N in N_list:

        #k_lin_pos = np.arange(N/2+1,10001)
        #k_lin_neg = np.arange(-10000,-(N/2+1)+1)
        #trunc_err.append(2*np.pi*sum(np.abs(uk_func(np.concatenate([k_lin_neg,k_lin_pos])))**2))

        k_lin = np.arange(-N/2,N/2+1)
        x_lin = np.linspace(0,2*np.pi,N)

        u_approx = fourier_approx(k_lin,x_lin,uk_func(k_lin))

        trunc_err.append(np.max(np.abs((u_func(x_lin)-u_approx))))

    return trunc_err

def fourier_approx(k_lin,x_lin,uk):

    u_approx_func = lambda x: np.sum(uk*np.exp(1j*k_lin*x))
    u_approx = np.array([u_approx_func(x) for x in x_lin])

    return u_approx

def discrete_fourier_coefficients(k_lin,u_func=lambda x: 1/(2-np.cos(x))):

    N = len(k_lin)-1
    uk_approx   = np.zeros_like(k_lin,dtype=complex)

    for k_idx,k in enumerate(k_lin): 
        s = 0
        for j in range(N+1):
            xj = 2*np.pi*j/(N+1)
            s += u_func(xj)*np.exp(-1j*k*xj) 

        uk_approx[k_idx] = s/(N+1)

    return uk_approx

def match_uk(uk,uk_approx):

    return np.allclose(uk,uk_approx,rtol=0.1,atol=0.1)


if __name__ == "__main__":

    # Analytical function 
    u_func = lambda x: 1/(2-np.cos(x)) # Remove pi, to let x = [0:2pi]
    #u_func = lambda x: x
    
    # Fourier coefficients
    uk_func = lambda k: 1/(np.sqrt(3)*(2+np.sqrt(3))**np.abs(k))

    # Discrete fourier coefficients
    uk_approx_func = lambda k: discrete_fourier_coefficients(k,u_func=u_func)
    
    #uk_func = lambda k: np.where(k == 0, np.pi + 0j, 1j / k) # test kode eksempel fra slide

    # Initializations
    N = 64
    N_convergence_list = np.array([4,8,16,32,64,128])
    trunc_err = convergence_list(N_convergence_list,fourier_approx,u_func,uk_func)
    k_lin = np.arange(-N/2,N/2+1)
    
    # Convergence plot
    plt.figure(1)
    plt.semilogy(N_convergence_list,trunc_err)
    plt.xlabel("N")
    plt.ylabel(r"$||\tau||^2$")
    plt.title("Convergence plot")

    # Discrete fourier transform

    uk_approx = {} # Using dictionary
    for Ni in N_convergence_list: # Changing the size of N that is the number of waves

        k_lin_temp = np.arange(-Ni/2,Ni/2+1) 
        uk_approx[Ni] = discrete_fourier_coefficients(k_lin_temp,u_func) # Discrete fourier transformation
        match = match_uk(uk_func(k_lin_temp),uk_approx[Ni])      # Matching the discrete FT with the analytical
        print(match)

    # Plotting the analytical uk vs the approximated
    plt.figure(2)
    plt.plot(k_lin_temp,uk_approx[Ni],"o-",label=f"N={Ni}")
    plt.plot(k_lin_temp,uk_func(k_lin_temp),"o-",label="uk analytical")
    plt.xlabel("k")
    plt.ylabel("uk")
    plt.legend()

    Ni = 50
    x_lin = np.linspace(0,2*np.pi,Ni)
    k_lin_temp = np.arange(-Ni/2,Ni/2+1) 
    # Approximating function using fourier coefficients
    u_approx_analytical = fourier_approx(k_lin_temp,x_lin,uk_func(k_lin_temp))
    u_approx_discrete = fourier_approx(k_lin_temp,x_lin,uk_approx_func(k_lin_temp))

    # Plotting the approximated function values
    plt.figure(3)
    plt.plot(x_lin,u_approx_analytical.real,label="u analytical")
    plt.plot(x_lin,u_approx_discrete.real,label="u discrete")
    plt.legend()

    # Comparison of convergence using analytical and discrete
    trunc_err_approx = convergence_list(N_convergence_list,fourier_approx,u_func,uk_approx_func)

    plt.figure(4)
    plt.semilogy(N_convergence_list,trunc_err,label="Analytical")
    plt.semilogy(N_convergence_list,trunc_err_approx,label="Discrete")
    plt.xlabel("N")
    plt.ylabel(r"$||\tau||^2$")
    plt.legend()
    plt.title("Convergence plot")

    plt.show()



