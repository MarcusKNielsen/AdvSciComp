
#%%
import numpy as np
import func.legendre as legendre
from advection_v2 import f_func, g_func,total_grid_points
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from time import perf_counter

# Parameters
x_left = -1
x_right = 1
a = 1.0 
alpha = 0.0
#max_step = 0.00001 gamle med 1 orden for lidt
#tf = 0.0001 gamle med 1 orden for lidt
max_step = 0.0001
#tf = 1.0
tf = 0.1

formulation = "w"

def g0_val(t):
    return np.sin(np.pi*(-1-a*t))

# Convergence test 
N_list = np.arange(2,8) 
K_list = np.logspace(1,2,10,dtype=int)

error = np.zeros([len(N_list),len(K_list)])
times = np.zeros([len(N_list),len(K_list)])

for N_idx,N in enumerate(N_list):
    for K_idx,K in enumerate(K_list):

        x_nodes = legendre.nodes(N)
        x_total = total_grid_points(K,x_nodes,x_left,x_right)
        u0 =  np.sin(np.pi*x_total)
        h = (x_total[-1]-x_total[0])/K
        
        V,Vx,w = legendre.vander(x_nodes)
        M = np.linalg.inv(V@V.T)
        Mk = (h/2)*M
        Mk_inv = np.linalg.inv(Mk) 
        Dx = Vx@np.linalg.inv(V)
        S = M@Dx

        t_start = perf_counter()
        sol = solve_ivp(f_func, [0, tf], u0, args=(Mk_inv,S,N,alpha,a,g0_val,formulation), max_step=max_step, dense_output=True, method="RK23")
        t_slut = perf_counter()-t_start     


        err = sol.y[:,-1] - g_func(x_total,tf,a)
        I = np.eye(K)
        M_total = np.kron(I,Mk)
        error[N_idx,K_idx] = np.sqrt(err @ M_total @ err)
        times[N_idx,K_idx] = np.round(t_slut,2)
       

#%%

plt.figure()
plt.title("Convergence Test (Advection)")

convergence_rate = np.zeros(len(N_list))
for N_idx,N in enumerate(N_list):
    
    plt.loglog(K_list,error[N_idx],"--o",label=f"N={N-1}")

    # Fit a first-order polynomial (line)
    coefficients = np.polyfit(np.log(K_list)[:-1], np.log(error[N_idx])[:-1], 1)

    # Extract slope (a) and intercept (b)
    a, b = coefficients

    #plt.plot(K_list,-N*K_list+b,label=f"N+1={N}")
    
    print(f"N = {N}")
    print(f"a = {a}")
    #print(f"Intercept (b): {b}")
    convergence_rate[N_idx] = N #np.abs(a)

plt.xticks(K_list[::2], labels=K_list[::2])
plt.xlabel(r"$K$: Number of Elements")
plt.ylabel(r"$\Vert u - u_h \Vert_{L^2}$")
plt.legend()

# Display results as a table
print("N \\ K", *K_list, "Convergence rate", sep="\t")
for N_idx, N in enumerate(N_list):
    row = [f"{error[N_idx, K_idx]:.1E}" if error[N_idx, K_idx] != 0 else "-" for K_idx in range(len(K_list))]
    rate = f"{convergence_rate[N_idx]:.1f}"
    print(f"{N-1}\t" + "\t".join(row) + f"\t{rate}")
    
# Display results as a table
print("N \\ K", *K_list, "Convergence rate", sep="\t")
for N_idx, N in enumerate(N_list):
    row = [f"{times[N_idx, K_idx]:.1E}" if times[N_idx, K_idx] != 0 else "-" for K_idx in range(len(K_list))]
    rate = f"{convergence_rate[N_idx]:.1f}"
    print(f"{N-1}\t" + "\t".join(row) + f"\t{rate}")

plt.show()

