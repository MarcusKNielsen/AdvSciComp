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

        trunc_err.append(np.max(np.abs((u_func(x_lin)-u_approx))))

    return trunc_err

def fourier_approx(k_lin,x_lin,uk):

    u_approx_func = lambda x: np.sum(uk*np.exp(1j*k_lin*x))
    u_approx = np.array([u_approx_func(x) for x in x_lin])

    return u_approx

def discrete_fourier_coefficients(k_lin,u_func=lambda x: 1/(2-np.cos(x))):

    N = len(k_lin) 
    uk_approx   = np.zeros_like(k_lin,dtype=complex)

    if (N%2) == 0:

        for k_idx,k in enumerate(k_lin): 
            s = 0

            ck = lambda ck_k: 2 if np.abs(ck_k) == N/2 else 1

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

def diff_v(x):
    # The analytical derivative of the function v in exercise d
    return np.cos(x)*np.exp(np.sin(x))

# Analytical function 
u_func = lambda x: 1/(2-np.cos(x)) # Remove pi, to let x = [0:2pi]

# Fourier coefficients
uk_func = lambda k: 1/(np.sqrt(3)*(2+np.sqrt(3))**np.abs(k))

# Discrete Fourier coefficients
uk_approx_func = lambda k: discrete_fourier_coefficients(k,u_func=u_func)


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
plt.ylabel(r"$||\tau||^2$")
plt.legend(fontsize=12) 
plt.title("Convergence Plot (logarithmic y-axis)")
#plt.show()

#%%

"""
Exercise 1: b)
Comparison of analytical and discrete fourier coefficients for different N
"""

# Define fontsize variable
fontsize = 12

# N = 4
N1 = 4
k_lin1 = check_N(N1)

uk_approx1 = discrete_fourier_coefficients(k_lin1).real
uk_exact1 = uk_func(k_lin1)

# N = 10
N2 = 10
k_lin2 = check_N(N2)

uk_approx2 = discrete_fourier_coefficients(k_lin2).real
uk_exact2 = uk_func(k_lin2)

# Create a 2x2 grid for the subplots
plt.figure(1)
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# First subplot (N = 4, comparison of Fourier coefficients)
axs[0, 0].plot(k_lin1, np.abs(uk_approx1 - uk_exact1), "o-")
axs[0, 0].set_xlabel("k", fontsize=fontsize)
axs[0, 0].set_ylabel(r"$|\tilde{u}_k - \hat{u}_k|$", fontsize=fontsize)
axs[0, 0].set_title(r"Error with $N=4$", fontsize=fontsize)

# Second subplot (N = 4, comparison of approx and exact)
axs[0, 1].plot(k_lin1, uk_approx1, "o-", label=r"Approx: $\tilde{u}_k$")
axs[0, 1].plot(k_lin1, uk_exact1, "o-", label=r"Exact: $\hat{u}_k$")
axs[0, 1].set_xlabel("k", fontsize=fontsize)
axs[0, 1].set_ylabel(r"$u_k$", fontsize=fontsize)
axs[0, 1].legend(fontsize=fontsize)
axs[0, 1].set_title(r"Approx vs Exact with $N=4$", fontsize=fontsize)

# Third subplot (N = 10, comparison of Fourier coefficients)
axs[1, 0].plot(k_lin2, np.abs(uk_approx2 - uk_exact2), "o-")
axs[1, 0].set_xlabel("k", fontsize=fontsize)
axs[1, 0].set_ylabel(r"$|\tilde{u}_k - \hat{u}_k|$", fontsize=fontsize)
axs[1, 0].set_title(r"Error with $N=10$", fontsize=fontsize)

# Fourth subplot (N = 10, comparison of approx and exact)
axs[1, 1].plot(k_lin2, uk_approx2, "o-", label=r"Approx: $\tilde{u}_k$")
axs[1, 1].plot(k_lin2, uk_exact2, "o-", label=r"Exact: $\hat{u}_k$")
axs[1, 1].set_xlabel("k", fontsize=fontsize)
axs[1, 1].set_ylabel(r"$u_k$", fontsize=fontsize)
axs[1, 1].legend(fontsize=fontsize)
axs[1, 1].set_title(r"Approx vs Exact with $N=10$", fontsize=fontsize)

# Adjust layout to avoid overlap
plt.tight_layout()

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
    

plt.figure(2)
plt.plot(N_list,np.log(err),"o-",label=r"$\max_k \ |\tilde{u}_k - \hat{u}_k|$")
plt.xlabel("N")
plt.ylabel(r"Error")
plt.title("Convergence of Fourier Coefficients")
plt.legend(fontsize=12)
#plt.show()


#%%

if __name__ == "__main__":


    # Initializations
    N = 64
    N_convergence_list = np.arange(4,128,4)
    trunc_err = convergence_list(N_convergence_list,fourier_approx,u_func,uk_func)
    k_lin = check_N(N)
    
    # Convergence plot
    plt.figure(3)
    plt.semilogy(N_convergence_list,trunc_err,"o-",label=r"Numerical: $||u - P_Nu ||^2$")
    #plt.semilogy(N_convergence_list[:-17],norm_2_tau(N_convergence_list[:-17]),label=r"Analytical: $||\tau||^2 \sim e^{- \alpha \frac{N}{2}}$")
    plt.xlabel("N")
    plt.ylabel(r"$||\tau||^2$")
    plt.legend()
    plt.title("Convergence Plot (Semilog)")

    # Discrete fourier transform

    uk_approx = {} # Using dictionary
    for Ni in N_convergence_list: # Changing the size of N that is the number of waves

        k_lin_temp = check_N(Ni) 
        uk_approx[Ni] = discrete_fourier_coefficients(k_lin_temp,u_func) # Discrete fourier transformation
        match = match_uk(uk_func(k_lin_temp),uk_approx[Ni])      # Matching the discrete FT with the analytical

    # Plotting the analytical uk vs the approximated
    plt.figure(4)
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
    plt.figure(5)
    plt.plot(x_lin,u_approx_analytical.real,label="u analytical")
    plt.plot(x_lin,u_approx_discrete.real,label="u discrete")
    plt.legend()

    # Comparison of convergence using analytical and discrete
    trunc_err_approx = convergence_list(N_convergence_list,fourier_approx,u_func,uk_approx_func)

    plt.figure(6)
    plt.semilogy(N_convergence_list,trunc_err,"o-",label=r"Numerical: $||u - P_Nu ||^2$")
    plt.semilogy(N_convergence_list,trunc_err_approx,"o-",label="Discrete")
    plt.semilogy(N_list[:-17],(2-np.sqrt(3))**(N_list[:-17]/2),label=r"Analytical: $||\tau||^2 \sim e^{- \alpha \frac{N}{2}}$")
    plt.xlabel("N")
    plt.ylabel(r"$||\tau||^2$")
    plt.legend()
    plt.title("Convergence plot")

    #%% Lagrange interpolation Exercise C

    # Initializations
    k_lin   = check_N(10)
    N       = len(k_lin)
    j_lin   = np.arange(0,N)
    xj      = 2*np.pi*j_lin/(N)

    x = np.linspace(0,2*np.pi,1000) 

    # Visualizing the lagrange polynomials
    plt.figure(7)

    for j_idx in range(N):
        
        y = lagrange_interpolation(xj[j_idx],x,N)
        plt.plot(x,y,label="$j_{idx}$"+f"={j_idx}")

        pointx = xj
        pointy = lagrange_interpolation(xj[j_idx],xj,N)
        plt.scatter(pointx,pointy,label="$j_{idx}$"+f"={j_idx}")

    plt.title("Lagrange polynomials")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    plt.tight_layout()

    Dh = lambda xj, x, N: np.where(
        np.abs(x - xj) < 1e-1, 
        0, 
        (N * np.cos(N / 2 * (x - xj)) * np.cos((x / 2) - (xj / 2)) * np.sin((x / 2) - (xj / 2)) - 
        np.sin(N / 2 * (x - xj)) /
        (2 * N * (np.sin((x / 2) - (xj / 2))**2)))
    )

    plt.figure(8)
    plt.plot(x,Dh(xj[2],x,N),label="dh/dx")
    plt.plot(x,10*lagrange_interpolation(xj[2],x,N),label="h")
    plt.legend()
    #plt.scatter(xj,Dh(xj[2],xj,N))


    D = np.zeros([N,N])
    for j in j_lin:
        D[:,j] = Dh(xj[j],xj,N)

    Dv_approx = D@v(xj)

    Dv_exact = diff_v(xj)    

    plt.figure(9)
    plt.plot(xj,Dv_exact,label="exact")
    plt.plot(xj,Dv_approx,label="approx")
    plt.grid()
    plt.show()

    debug = True

