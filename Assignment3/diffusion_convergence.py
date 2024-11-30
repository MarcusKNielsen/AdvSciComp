import numpy as np
import func.legendre as legendre
from diffusion_v2 import f_func,u_exact,total_grid_points
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
x_left = -1
x_right = 1
a = 1.0 
alpha = 1.0
max_step = 0.0001
t0 = 0.005
tf = 0.006

# Convergence test 
N_list = np.arange(3,11) 
K_list = np.logspace(1,2,20,dtype=int)

error = np.zeros([len(N_list),len(K_list)])

for N_idx,N in enumerate(N_list):
    for K_idx,K in enumerate(K_list):

        x_nodes = legendre.nodes(N)
        x_total = total_grid_points(K,x_nodes,x_left,x_right)

        h = (x_total[-1]-x_total[0])/K

        V,Vx,_ = legendre.vander(x_nodes)
        M = np.linalg.inv(V@V.T)
        Mk = (h/2)*M
        Mk_inv = np.linalg.inv(Mk) 
        Dx = Vx@np.linalg.inv(V)
        S = M@Dx

        u0 = u_exact(x_total,t0,a)

        sol = solve_ivp(f_func, [t0, tf], u0, args=(Mk_inv,Dx,S,N,alpha,a), max_step=max_step, dense_output=True, method="Radau")
        
        err = sol.y[:,-1] - u_exact(x_total,tf,a)
        I = np.eye(K)
        M_total = np.kron(I,Mk)
        error[N_idx,K_idx] = np.sqrt(err @ M_total @ err)
        #error[N_idx,K_idx] = np.max(np.abs(u_exact(x_total,sol.t[-1],a) - sol.y[:,-1]))
        
#%%

plt.figure()
plt.title("Convergence Test (Diffusion)")

for N_idx,N in enumerate(N_list):
    
    plt.loglog(K_list,error[N_idx],"-o",label=f"N={N}")

    # Fit a first-order polynomial (line)
    coefficients = np.polyfit(np.log(K_list), np.log(error[N_idx]), 1)

    # Extract slope (a) and intercept (b)
    a, b = coefficients
    
    print(f"N = {N}")
    print(f"a = {a}")
    #print(f"Intercept (b): {b}")

plt.xticks(K_list[::2], labels=K_list[::2])
plt.xlabel(r"$K$: Number of Elements")
plt.ylabel(r"$\Vert u - u_h \Vert_{L^2}$")
plt.legend()
plt.show()

