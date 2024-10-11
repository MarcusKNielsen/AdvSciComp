import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,2*np.pi,1000)

def w0(x):
    x = 2*x-2*np.pi
    return (x < 0)*(-np.cos(x)) + (x >= 0)*np.cos(x)

def w1(x):
    x = 2*x-2*np.pi
    return np.sin(np.maximum(x,0)) - np.sin(np.minimum(x,0))

def w2(x):
    x = 2*x-2*np.pi
    return -np.cos(np.maximum(x,0)) + np.cos(np.minimum(x,0))

def w3(x):
    x = 2*x-2*np.pi
    return - np.sin(np.abs(x)) + np.abs(x) - 2*np.pi

plt.figure()
plt.plot(x,w0(x),label=r"$w^0$")
plt.plot(x,w1(x),label=r"$w^1$")
plt.plot(x,w2(x),label=r"$w^2$")
plt.plot(x,w3(x),label=r"$w^3$")
plt.xlabel(r"$x$")
plt.legend()
plt.show()

def w(x,i):
    if i == 0:
        return w0(x)
    elif i == 1:
        return w1(x)
    elif i == 2:
        return w2(x)
    elif i == 3:
        return w3(x)
    
def check_N(N):

    if (N%2) == 0: 
        N_even = N 
        k_lin = np.arange(-N_even/2,N_even/2)
    else:
        N_odd = N - 1 
        k_lin = np.arange(-N_odd/2,N_odd/2+1)

    return k_lin

def D_matrix(N, xj, j_lin):
    def Dh(xj_scalar, x_array, N):
        delta = x_array - xj_scalar
        small_tol = 1e-14  # A very small number to avoid division by zero
        
        # Compute numerator and denominator separately
        numerator = (np.cos(N * delta / 2) * np.cos(delta / 2) * N * np.sin(delta / 2) - np.sin(N * delta / 2))
        denominator = 4 * N * (np.sin(delta / 2))**2
        
        # Initialize result array with zeros
        result = np.zeros_like(delta)
        
        # Perform safe division where denominator is non-zero
        result = np.divide(numerator, denominator, out=result, where=np.abs(denominator) > small_tol)
        
        # Handle the case where delta == 0
        # Since both numerator and denominator approach zero, the limit is zero
        result[np.abs(delta) <= small_tol] = 0
        
        return result

    D = np.zeros((N, N))
    for idx, j in enumerate(j_lin):
        D[:, idx] = Dh(xj[j], xj, N)
        
    return D, Dh


def discrete_inner_product(u,v):
    N = len(u)
    return (2*np.pi/N)*np.sum(u * np.conjugate(v))

def discrete_norm(u):
    return discrete_inner_product(u,u)

N = 8
k_lin = check_N(N)

j_lin   = np.arange(0,N)
xj      = 2*np.pi*j_lin/N
D,_ = D_matrix(N,xj,j_lin)

i = 1
Dv_approx = D@w(xj,i)

plt.figure()
plt.plot(xj,Dv_approx,".-")
plt.plot(x,w(x,i-1),label=r"$w^0$")
plt.xlabel("x")

plt.show()


#%%

plt.figure(11)
for i in range(1,4):
    N_convergence_list = np.arange(4,256,16)
    err = np.zeros(len(N_convergence_list))
    for idx,Nc in enumerate(N_convergence_list):
        j_lin_loop   = np.arange(0,Nc)
        xj_loop      = 2*np.pi*j_lin_loop/(Nc)
        D,_     = D_matrix(Nc,xj_loop,j_lin_loop)
        Dv_approx = D@w(xj_loop,i)
        Dv_exact = w(xj_loop,i-1) 
    
        err[idx] = discrete_norm(Dv_approx-Dv_exact)
    plt.loglog(N_convergence_list,err,"o-",label=fr"$\Vert \frac{{d w^{{{i}}}}}{{dx}} - D w^{{{i}}} \Vert_N$",markersize=4)

plt.loglog(N_convergence_list,  40 / N_convergence_list**1, linestyle='--', label=r'Decay$\sim N^{-1}$')
plt.loglog(N_convergence_list, 200 / N_convergence_list**3, linestyle='--', label=r'Decay$\sim N^{-3}$')
plt.loglog(N_convergence_list, 500 / N_convergence_list**5, linestyle='--', label=r'Decay$\sim N^{-5}$')

plt.xlabel("N")
plt.ylabel("$||v'(x)-Dv||_N$")
plt.title("Discrete $L^2$ error of first derivative (loglog plot)")
plt.grid()
plt.legend()
plt.show()










