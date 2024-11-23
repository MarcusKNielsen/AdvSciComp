import numpy as np
import func.legendre as legendre
from advec_diff import f_func,u_exact,total_grid_points
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
x_left = -1
x_right = 1
a = 1.0 
d = 0.5
alpha = 1.0
max_step = 0.01
t0 = 0.05
tf = 0.06
formulation = "s"

# Convergence test
N_list = np.arange(3,7)
K_list = np.logspace(1,2,10,dtype=int)

error = np.zeros([len(N_list),len(K_list)])

for N_idx,N in enumerate(N_list):
    for K_idx,K in enumerate(K_list):
        print(f"N = {N}, K = {K}")
        x_nodes = legendre.nodes(N)
        x_total = total_grid_points(K,x_nodes,x_left,x_right)
        u0 = u_exact(x_total,t0,a,d)
        h = (x_total[-1]-x_total[0])/K
        
        V,Vx,w = legendre.vander(x_nodes)
        M = np.linalg.inv(V@V.T)
        Mk = (h/2)*M
        Mk_inv = np.linalg.inv(Mk) 
        Dx = Vx@np.linalg.inv(V)
        S = M@Dx

        sol = solve_ivp(f_func, [t0, tf], u0, args=(Mk_inv,Dx,S,N,alpha,a,d,formulation), max_step=max_step, dense_output=True, method="Radau")
            
        error[N_idx,K_idx] = np.max(np.abs(u_exact(x_total,sol.t[-1],a,d) - sol.y[:,-1]))
        
#%%

plt.figure()
plt.title("Convergence Test")

for N_idx,N in enumerate(N_list):
    
    plt.loglog(K_list,error[N_idx],"-o",label=f"N={N}")

    # Fit a first-order polynomial (line)
    coefficients = np.polyfit(np.log(K_list), np.log(error[N_idx]), 1)

    # Extract slope (a) and intercept (b)
    a, b = coefficients
    
    print(f"N = {N}")
    print(f"a = {a}")
    #print(f"Intercept (b): {b}")

plt.xlabel(r"$K$: Number of Elements")
plt.ylabel(r"$\max|u - u_h|$")
plt.legend()
plt.show()

