#%%
import matplotlib.pyplot as plt
import numpy as np


def check_N(N):

    if (N%2) == 0: 
        N_even = N 
        k_lin = np.arange(-N_even/2,N_even/2)
    else:
        N_odd = N - 1 
        k_lin = np.arange(-N_odd/2,N_odd/2+1)

    return k_lin

def convergence_list(N_list,fourier_approx,u_func,uk_func):

    trunc_err = []

    for N in N_list:

        k_lin = check_N(N)
        x_lin = np.linspace(0,2*np.pi,N)

        u_approx = fourier_approx(k_lin,x_lin,uk_func(k_lin))

        trunc_err.append(np.max(np.abs(u_func(x_lin)-u_approx)))

    return trunc_err

def fourier_approx(k_lin,x_lin,uk):

    u_approx_func = lambda x: np.sum(uk*np.exp(1j*k_lin*x))
    u_approx = np.array([u_approx_func(x) for x in x_lin])

    return u_approx

def discrete_fourier_coefficients(k_lin,u_func=lambda x: 1/(2-np.cos(x))):

    N = len(k_lin) 
    uk_approx   = np.zeros_like(k_lin,dtype=complex)

    if (N%2) == 0:
        
        ck = lambda ck_k: 2 if np.abs(ck_k) == N/2 else 1
        
        for k_idx,k in enumerate(k_lin): 
            s = 0
            for j in range(N):
                xj = 2*np.pi*j/N 
                s += (u_func(xj)/ck(k))*np.exp(-1j*k*xj) 

            uk_approx[k_idx] = s/N
    else:

        for k_idx,k in enumerate(k_lin): 
            s = 0
            for j in range(N+1):
                xj = 2*np.pi*j/N
                s += u_func(xj)*np.exp(-1j*k*xj) 

            uk_approx[k_idx] = s/N

    return uk_approx

def lagrange_interpolation(xj,x,N):
    h_even = lambda xj, x: np.where(x == xj, 1, (1/N * np.sin(N/2 * (x-xj)) * (1/np.tan(1/2 * (x-xj)))))
    h_odd  = lambda xj, x: np.where(x == xj, 1, (1/N)*np.sin((x-xj)*N/2)/np.sin((x-xj)/2))
    h = lambda xj,x,N: h_even(xj,x) if (N%2)== 0 else h_odd(xj,x)

    return h(xj,x,N)

def match_uk(uk,uk_approx):

    return np.allclose(uk,uk_approx,rtol=0.1,atol=0.1)

def v(x):
    return np.exp(np.sin(x))
    #return np.sin(x)

def diff_v(x):
    # The analytical derivative of the function v in exercise d
    return np.cos(x)*np.exp(np.sin(x))
    #return np.cos(x)

def D_matrix(N,xj,j_lin):
    Dh = lambda xj,x,N: np.where(
        np.abs(x-xj) < 1e-12,
        0,
        (np.cos(N*(x - xj)/2)*np.cos(x/2 - xj/2)*N*np.sin(x/2 - xj/2) - np.sin(N*(x - xj)/2))/(2*np.sin(x/2 - xj/2)**2*N)
    )

    D = np.zeros([N,N])
    for j in j_lin:
        D[:,j] = Dh(xj[j],xj,N)
    
    return D,Dh

# Analytical function 
u_func = lambda x: 1/(2-np.cos(x)) # Remove pi, to let x = [0:2pi]

# Fourier coefficients
uk_func = lambda k: 1/(np.sqrt(3)*(2+np.sqrt(3))**np.abs(k))

# Discrete Fourier coefficients
uk_approx_func = lambda k: discrete_fourier_coefficients(k,u_func=u_func)





#%%

if __name__ == "__main__":

    #%%

    """
    Exercise 1: a)
    The section creates a convergence plot. Comparing the analytical and numerical expressions:
    Analytical: ||tau||² ~ exp(-aN/2) = (2-sqrt(3))**(N/2)
    Numerical:  ||tau||² = || u - P_N u||
    """

    N_list = np.arange(4,128,4)
    trunc_err = convergence_list(N_list,fourier_approx,u_func,uk_func)

    # Convergence plot
    plt.figure(1)
    plt.semilogy(N_list,trunc_err,"o-",label=r"Numerical: $||u - P_Nu ||^2$")
    plt.semilogy(N_list[:-17],(2-np.sqrt(3))**(N_list[:-17]/2),label=r"Analytical: $||\tau||^2 \sim e^{- \alpha \frac{N}{2}}$")
    plt.xlabel("N")
    plt.legend(fontsize=12) 
    #plt.show()


    #%%
    
    """
    Exercise 1: b)
    Comparison of analytical and discrete fourier coefficients for different N
    """
    
    # Define fontsize variable
    fontsize = 10
    
    # Define all the N values
    N_values = [4, 8, 16, 32, 64]
    
    # Create lists to store k_lin, uk_approx, and uk_exact for all N values
    k_lin_list = []
    uk_approx_list = []
    uk_exact_list = []
    
    # Loop through all N values to calculate the necessary arrays
    for N in N_values:
        k_lin = check_N(N)
        uk_approx = discrete_fourier_coefficients(k_lin).real
        uk_exact = uk_func(k_lin)
        
        k_lin_list.append(k_lin)
        uk_approx_list.append(uk_approx)
        uk_exact_list.append(uk_exact)
    
    # Create a 5x2 grid for the subplots (10 subplots total)
    fig, axs = plt.subplots(5, 2, figsize=(9, 9))  # Changed to 5x2 grid
    
    # Flatten axs for easy access
    axs = axs.flatten()
    
    # Loop through each N value and populate the subplots
    for i, N in enumerate(N_values):
        k_lin = k_lin_list[i]
        uk_approx = uk_approx_list[i]
        uk_exact = uk_exact_list[i]
        
        # First plot (Error comparison for N)
        axs[2*i].plot(k_lin, np.abs(uk_approx - uk_exact), "o-")
        if N == N_values[-1]:
            axs[2*i].set_xlabel("k", fontsize=fontsize)
        axs[2*i].set_ylabel(r"$|\tilde{u}_k - \hat{u}_k|$", fontsize=fontsize)
        if N == N_values[0]:
            axs[2*i].set_title(f"Error", fontsize=fontsize+2)
        
        # Add a text box in the top-left corner to indicate N value
        axs[2*i].text(0.6, 0.90, f'N = {N}', transform=axs[2*i].transAxes, 
                      fontsize=fontsize, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    
        # Second plot (Approx vs Exact for N)
        axs[2*i+1].semilogy(k_lin, uk_approx, "o-", label=r"Approx: $\tilde{u}_k$")
        axs[2*i+1].semilogy(k_lin, uk_exact, "o-", label=r"Exact: $\hat{u}_k$")
        if N == N_values[-1]:
            axs[2*i+1].set_xlabel("k", fontsize=fontsize)
        axs[2*i+1].set_ylabel(r"$u_k$", fontsize=fontsize)
        axs[2*i+1].legend(loc='lower center',fontsize=fontsize-1)
        if i == 0:
            axs[2*i+1].set_title(f"Approx vs Exact Coefficients", fontsize=fontsize+2)
    
        # Add a text box in the top-left corner to indicate N value
        axs[2*i+1].text(0.05, 0.90, f'N = {N}', transform=axs[2*i+1].transAxes, 
                        fontsize=fontsize, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    
    # Adjust layout to add more vertical space
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)  # Increase vertical space between plots
    
    # Display the plot
    #plt.show()




    #%%

    """
    Exercise 1: b)
    Convergence of fourier coefficients
    """

    N_list = np.arange(4,128,4)

    err = np.zeros_like(N_list,dtype=float)
    for i,Ni in enumerate(N_list): # Changing the size of N that is the number of waves

        k_lin = check_N(Ni)
        uk_approx = discrete_fourier_coefficients(k_lin,u_func).real # Discrete fourier transformation
        uk_exact  = uk_func(k_lin)
        err[i] = np.max(np.abs(uk_approx-uk_exact))
        

    plt.figure(3)
    plt.plot(N_list,np.log(err),"o-",label=r"$\max_k \ |\tilde{u}_k - \hat{u}_k|$")
    plt.xlabel("N")
    plt.title("Convergence of Fourier Coefficients")
    plt.legend(fontsize=12)
    #plt.show()


    # Initializations
    #N = 64
    N_convergence_list = np.arange(4,64*2,4)
    trunc_err = convergence_list(N_convergence_list,fourier_approx,u_func,uk_func)
    #k_lin = check_N(N)
    
    # Convergence plot
    plt.figure(4)
    plt.semilogy(N_convergence_list,trunc_err,"o-",label=r"Numerical: $||u - P_Nu ||^2$")
    #plt.semilogy(N_convergence_list[:-17],norm_2_tau(N_convergence_list[:-17]),label=r"Analytical: $||\tau||^2 \sim e^{- \alpha \frac{N}{2}}$")
    plt.xlabel("N")
    plt.legend()
    
    
    # Discrete fourier transform

    uk_approx = {} # Using dictionary
    for Ni in N_convergence_list: # Changing the size of N that is the number of waves

        k_lin_temp = check_N(Ni) 
        uk_approx[Ni] = discrete_fourier_coefficients(k_lin_temp,u_func) # Discrete fourier transformation
        match = match_uk(uk_func(k_lin_temp),uk_approx[Ni])      # Matching the discrete FT with the analytical

    # Plotting the analytical uk vs the approximated
    plt.figure(5)
    plt.plot(k_lin_temp,uk_approx[Ni],"o-",label=f"N={Ni}")
    plt.plot(k_lin_temp,uk_func(k_lin_temp),"o-",label="uk analytical")
    plt.xlabel("k")
    plt.ylabel("uk")
    plt.legend()

    Ni = 10
    x_lin = np.linspace(0,2*np.pi,Ni)
    k_lin_temp = check_N(Ni)
    # Approximating function using fourier coefficients
    u_approx_analytical = fourier_approx(k_lin_temp,x_lin,uk_func(k_lin_temp))
    u_approx_discrete = fourier_approx(k_lin_temp,x_lin,uk_approx_func(k_lin_temp))

    # Plotting the approximated function values
    plt.figure(6)
    plt.plot(x_lin,u_approx_analytical.real,label="u analytical")
    plt.plot(x_lin,u_approx_discrete.real,label="u discrete")
    plt.legend()

    # Comparison of convergence using analytical and discrete
    trunc_err_approx = convergence_list(N_convergence_list,fourier_approx,u_func,uk_approx_func)

    plt.figure(7)
    plt.semilogy(N_convergence_list,trunc_err,"o-",label=r"$||u - P_Nu ||^2$")
    plt.semilogy(N_convergence_list,trunc_err_approx,"o-",label="$||u - I_Nu||^2$")
    plt.semilogy(N_list[:-17],(2-np.sqrt(3))**(N_list[:-17]/2),label=r"$||\tau||^2 \sim e^{- \alpha \frac{N}{2}}$")
    plt.xlabel("N")
    plt.legend()

    #%% Lagrange interpolation Exercise C

    # Initializations
    k_lin   = check_N(6)
    N       = len(k_lin)
    j_lin   = np.arange(0,N)
    xj      = 2*np.pi*j_lin/(N)

    x_lin = np.linspace(0,2*np.pi,100) 

    # Visualizing the lagrange polynomials
    plt.figure(8)

    for j_idx in range(N):
        
        y = lagrange_interpolation(xj[j_idx],x_lin,N)
        plt.plot(x_lin,y,label=rf"$h_{{{j_idx}}}(x)$")

        pointx = xj
        pointy = lagrange_interpolation(xj[j_idx],xj,N)
        plt.scatter(pointx,pointy,label=rf"$h_{{{j_idx}}}(x_i) = \delta_{{{j_idx},i}}$")

    plt.xlabel("x")
    plt.ylabel("$h_j$(x)")
    plt.title("Lagrange polynomials")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
    plt.tight_layout()

    D,Dh = D_matrix(N,xj,j_lin)

    plt.figure(9)
    plt.plot(x_lin,Dh(xj[2],x_lin,N),label="dh/dx")
    plt.plot(x_lin,10*lagrange_interpolation(xj[2],x_lin,N),label="h")
    plt.legend()


    Dv_approx = D@v(xj)
    Dv_exact = diff_v(xj)    

    plt.figure(10)
    plt.plot(xj,Dv_exact,label="exact")
    plt.plot(xj,Dv_approx,label="approx")
    plt.grid()

    #%%

    # Plot 10
    Dv_exact = diff_v(x_lin)   
    plt.figure(11)
    plt.plot(x_lin,Dv_exact,label="v'(x)")



    # Plot approximate diff for several N and the convergence plot

    N_convergence_list = np.arange(4,64*2,4)
    err = np.zeros(len(N_convergence_list))
    for idx,Nc in enumerate(N_convergence_list):
        j_lin_loop   = np.arange(0,Nc)
        xj_loop      = 2*np.pi*j_lin_loop/(Nc)
        D,_     = D_matrix(Nc,xj_loop,j_lin_loop)
        Dv_approx = D@v(xj_loop) 
        Dv_exact = diff_v(xj_loop)  
        
        
        err[idx] = np.max(np.abs(Dv_approx-Dv_exact))

        # Add on plot 9
        Dv_approx = np.concatenate((Dv_approx, [Dv_approx[0]]))
        xj_loop = np.concatenate((xj_loop , [2*np.pi]))
        plt.plot(xj_loop,Dv_approx,label=f"$Dv$ for N = {Nc}",linestyle='--')
        plt.xlabel("x")
        plt.legend()
        plt.grid()
        

        
    plt.figure(12)
    plt.semilogy(N_convergence_list,err,".-",label="$||v'(x)-Dv||_\infty$")
    plt.xlabel("N")
    plt.legend()
    plt.grid()



    #%% Opgave f start ----------------------------------------------------------

    #%% Fast fourier transform

    from scipy.fftpack import fft, ifft

    k_lin_FFt = np.fft.fftfreq(N, d=(2 * np.pi) / N) * 2 * np.pi
    dvdx = ifft(1j*k_lin_FFt*fft(v(xj))).real
    dvdx = np.append(dvdx,dvdx[0])
    D,Dh = D_matrix(N,xj,j_lin)
    Dv_approx = D@v(xj)
    Dv_approx = np.append(Dv_approx, Dv_approx[0])
    Dv_exact = diff_v(x_lin)  
    xj2 = np.append(xj,np.array([2*np.pi]))

    plt.figure(13)
    plt.plot(x_lin,Dv_exact,label=r"$\frac{dv}{dx}$")
    plt.plot(xj2,Dv_approx,'-',label=r"$Dv$")
    plt.plot(xj2,dvdx,'--',label=r"FFT: $Dv$")
    plt.legend()
    plt.xlabel("x")
    plt.title(f"Using FFT to compute discrete derivative: N={N}")
    plt.grid()

    #%% FFT performance study 


    from time import perf_counter

    times_FFT = []
    times_Mat = []
    err_FFT = []

    #N_convergence_list = np.arange(4,64*20,10)

    for Nc in N_convergence_list:

        # Initializations for both methods
        k_lin_FFt = np.fft.fftfreq(Nc, d=(2 * np.pi) / Nc) * 2 * np.pi
        j_lin_loop   = np.arange(0,Nc)
        xj_loop      = 2*np.pi*j_lin_loop/(Nc)
        D,Dh = D_matrix(Nc,xj_loop,j_lin_loop)

        # Timing FFT 
        t_FFT = perf_counter()
        dvdx = ifft(1j*k_lin_FFt*fft(v(xj_loop))).real
        t_FFT = perf_counter()-t_FFT

        # Timing matrix 
        t_Dv = perf_counter()
        Dv_approx = D@v(xj_loop)
        t_Dv = perf_counter()- t_Dv

        times_FFT.append(t_FFT)
        times_Mat.append(t_Dv)

        # Saving the err to use for convergence
        Dv_exact = diff_v(xj_loop)
        err_FFT.append(np.max(np.abs(dvdx-Dv_exact)))


    plt.figure(14)
    plt.semilogy(N_convergence_list,err,".-",label="Matrix convergence")
    plt.semilogy(N_convergence_list,err_FFT,".-",label="FFT convergence")
    plt.xlabel("N")
    plt.legend()
    plt.grid()
    plt.show()

    # Performance study plot
    plt.figure(15)
    plt.semilogy(N_convergence_list,times_FFT,label="FFT")
    plt.semilogy(N_convergence_list,times_Mat,label="Mat")
    #plt.semilogy(N_convergence_list,N_convergence_list*np.log(N_convergence_list)*0.001,label="$N\logN$")
    #plt.semilogy(N_convergence_list,N_convergence_list**2*0.001,label="$N^2$")
    plt.xlabel("N")
    plt.ylabel("Time [seconds]")
    plt.legend()


    plt.show()

    #debug = True

