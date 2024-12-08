#%%
import numpy as np
import func.legendre as legendre
from diffusion_v2 import f_func,u_exact,total_grid_points
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from time import perf_counter

# Parameters
x_left = -1
x_right = 1
a = 1.0 
alpha = 1.0
max_step = 0.0001
t0 = 0.008
tf = 0.009

# Convergence test 
N_list = np.arange(3,12) 
K_list = np.logspace(1,2.5,20,dtype=int)

error = np.zeros([len(N_list),len(K_list)])
times = np.zeros([len(N_list),len(K_list)])

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

        t_start = perf_counter()
        sol = solve_ivp(f_func, [t0, tf], u0, args=(Mk_inv,Dx,S,N,alpha,a,"s"), max_step=max_step, dense_output=True, method="Radau")
        t_slut = perf_counter() - t_start

        err = sol.y[:,-1] - u_exact(x_total,tf,a)
        I = np.eye(K)
        M_total = np.kron(I,Mk)
        error[N_idx,K_idx] = np.sqrt(err @ M_total @ err)
        times[N_idx,K_idx] = t_slut

#%%

plt.figure()
plt.title("Convergence Test (Diffusion)")
convergence_rate = np.zeros(len(N_list))
for N_idx,N in enumerate(N_list):
    
    plt.loglog(K_list,error[N_idx],"--o",label=f"N={N-1}")

    # Fit a first-order polynomial (line)
    coefficients = np.polyfit(np.log(K_list), np.log(error[N_idx]), 1)

    # Extract slope (a) and intercept (b)
    a, b = coefficients
    
    print(f"N = {N-1}")
    print(f"a = {a}")
    #print(f"Intercept (b): {b}")
    convergence_rate[N_idx] = np.abs(a)


colors = plt.cm.tab10.colors

plt.loglog(K_list, 0.1*K_list.astype(float)**(-1), color = colors[0], label=rf"$\mathcal{{O}}(h^{{{1}}})$")
plt.loglog(K_list, 2.5*K_list.astype(float)**(-3), color = colors[2], label=rf"$\mathcal{{O}}(h^{{{3}}})$")
plt.loglog(K_list, 5.5*K_list.astype(float)**(-5), color = colors[4], label=rf"$\mathcal{{O}}(h^{{{5}}})$")
plt.loglog(K_list, 9.5*K_list.astype(float)**(-7), color = colors[6], label=rf"$\mathcal{{O}}(h^{{{7}}})$")

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


# %%

plt.figure(figsize=(6, 4))  # Adjust figure size if needed
plt.title("Convergence Test (Diffusion)")
convergence_rate = np.zeros(len(N_list))
for N_idx, N in enumerate(N_list):
    plt.loglog(K_list, error[N_idx], "--o", label=f"N={N-1}")

    coefficients = np.polyfit(np.log(K_list), np.log(error[N_idx]), 1)
    a, b = coefficients
    print(f"N = {N-1}")
    print(f"a = {a}")
    convergence_rate[N_idx] = np.abs(a)

colors = plt.cm.tab10.colors
plt.loglog(K_list, 0.1*K_list.astype(float)**(-1), color=colors[0], label=rf"$\mathcal{{O}}(h^{{{1}}})$")
plt.loglog(K_list, 2.5*K_list.astype(float)**(-3), color=colors[2], label=rf"$\mathcal{{O}}(h^{{{3}}})$")
plt.loglog(K_list, 5.5*K_list.astype(float)**(-5), color=colors[4], label=rf"$\mathcal{{O}}(h^{{{5}}})$")
plt.loglog(K_list[:14], 9.5*K_list[:14].astype(float)**(-7), color=colors[6], label=rf"$\mathcal{{O}}(h^{{{7}}})$")

plt.xticks(K_list[::2], labels=K_list[::2])
plt.xlabel(r"$K$: Number of Elements")
plt.ylabel(r"$\Vert u - u_h \Vert_{L^2}$")

plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Place legend outside
plt.subplots_adjust(right=0.8)  # Adjust space on the right to make room for the legend

# Display results as a table
print("N \\ K", *K_list, "Convergence rate", sep="\t")
for N_idx, N in enumerate(N_list):
    row = [f"{error[N_idx, K_idx]:.1E}" if error[N_idx, K_idx] != 0 else "-" for K_idx in range(len(K_list))]
    rate = f"{convergence_rate[N_idx]:.1f}"
    print(f"{N-1}\t" + "\t".join(row) + f"\t{rate}")

print("N \\ K", *K_list, "Convergence rate", sep="\t")
for N_idx, N in enumerate(N_list):
    row = [f"{(times[N_idx, K_idx]/times[0, 0]):.2f}" if times[N_idx, K_idx] != 0 else "-" for K_idx in range(len(K_list))]
    rate = f"{convergence_rate[N_idx]:.1f}"
    print(f"{N-1}\t" + "\t".join(row) + f"\t{rate}")

plt.show()

