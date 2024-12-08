#%% modules
import numpy as np
#import func.L2space as L2space
import func.legendre as legendre
import matplotlib.pyplot as plt
import advection_v2 
import diffusion_v2 


def scalability(method,order=1):
    #% Checking only for 100 elements and N
    K_list_test = np.arange(2,110,10)
    Eig_vec_test = np.zeros(len(K_list_test))
    for idx,ki in enumerate(K_list_test):
        A_ad_test = setup_A(12,ki,t,a,alpha_central,x_left,x_right,method)
        eigvals_ad_test = np.linalg.eigvals(A_ad_test)
        Eig_vec_test[idx] = np.max(np.abs(eigvals_ad_test))

    plt.figure()
    plt.plot(K_list_test,Eig_vec_test,"--o",label="$\max{|\lambda(\mathcal{A})|}$")
    # Fit a first-order polynomial (line)
    coefficients = np.polyfit(K_list_test, Eig_vec_test, order)
    fit = np.poly1d(coefficients)
    plt.plot(K_list_test,fit(K_list_test),label="Linear fit")
    if "advection" in str(method):
        plt.title("Scalability of advection scheme with variying $K$ and fixed $N=12$")
    if "diffusion" in str(method):
        plt.title("Scalability of diffusion scheme with variying $K$ and fixed $N=12$")
    plt.legend()
    plt.xlabel("K")
    plt.ylabel("$\max{|\lambda(\mathcal{A})|}$")
    # Extract slope (a) and intercept (b)
    print("coef:",coefficients)

def plt_eig(Eig_mat):
    plt.figure()
    N, K = np.meshgrid(N_list, K_list)

    # Create the pcolormesh plot
    pcm = plt.pcolormesh(K, N, Eig_mat.T)

    # Label the axes and add a title
    plt.xlabel("k: number of elements")
    plt.ylabel("N: number of nodes")
    plt.title("maximum eigenvalues")

    # Add the colorbar
    plt.colorbar(pcm, label="max|$\lambda$|")


def plt_eig_2d(Eig_mat):
    # Assuming Eig_mat, N_list, and number_element_list are defined
    N, K = np.meshgrid(N_list, K_list)

    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create the surface plot
    surf = ax.plot_surface(K, N, Eig_mat.T, cmap="viridis")

    # Label the axes and add a title
    ax.set_xlabel("k: number of elements")
    ax.set_ylabel("N: number of nodes")
    ax.set_zlabel("max|$\lambda$|")
    ax.set_title("Maximum Eigenvalues")

    # Add the colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="max|$\lambda$|")



def eig_scale(Eig_vec,K_list,order=2):
    plt.figure()
    plt.plot(K_list,Eig_vec,label="eigvals")
    # Fit a first-order polynomial (line)
    coefficients = np.polyfit(K_list, Eig_vec, order)
    fit = np.poly1d(coefficients)

    plt.plot(K_list,fit(K_list),label="fit")
    plt.legend()
    # Extract slope (a) and intercept (b)
    print("coef:",coefficients)

#%% Initializations

# Does not change
x_left = -1
x_right = 1
a = 1
t = 0

alpha_upwind  = 0.0
alpha_central = 1.0

#%% Eigen values test

def setup_A(N,K,t,a,alpha,x_left,x_right,method):

    x_nodes = legendre.nodes(N)

    x_total = method.total_grid_points(K,x_nodes,x_left,x_right)
    h = (x_total[-1]-x_total[0])/K

    V,Vx,_ = legendre.vander(x_nodes)
    M = np.linalg.inv(V@V.T)
    Mk = (h/2)*M
    Mk_inv = np.linalg.inv(Mk) 
    Dx = Vx@np.linalg.inv(V)
    S = M@Dx

    A = np.zeros([N*K,N*K])

    for i in range(N*K):
        e = np.zeros(N*K)
        e[i] = 1

        if method == advection_v2:
            A[:,i] = method.f_func(t,e,Mk_inv,S,N,alpha,a,method.g0_val,"w")
        else: 
            A[:,i] = method.f_func(t,e,Mk_inv,Dx,S,N,alpha,a,"s")
    
    return A

# Looping values
N_list = np.arange(2,20,2)
K_list = np.arange(2,20,2)

Eig_mat_ad = np.zeros([len(N_list),len(K_list)])
Eig_vec_ad = np.zeros(len(N_list))
Eig_mat_diff = np.zeros([len(N_list),len(K_list)])
Eig_vec_diff = np.zeros(len(N_list))

for N_idx,N in enumerate(N_list):
    for K_idx,K in enumerate(K_list):

        A_ad = setup_A(N,K,t,a,alpha_central,x_left,x_right,advection_v2)
        eigvals_ad = np.linalg.eigvals(A_ad)
        Eig_mat_ad[N_idx,K_idx] = np.max(np.abs(eigvals_ad))

        if N == K:
            Eig_vec_ad[N_idx] = np.max(np.abs(eigvals_ad))

        A_diff = setup_A(N,K,t,a,alpha_central,x_left,x_right,diffusion_v2)
        eigvals_diff = np.linalg.eigvals(A_diff)
        Eig_mat_diff[N_idx,K_idx] = np.max(np.abs(eigvals_diff))

        if N == K:
            Eig_vec_diff[N_idx] = np.max(np.abs(eigvals_diff))


# scalability(advection_v2)
# scalability(diffusion_v2,order=2)


#% Checking only for 100 elements and N
K_list_test = np.logspace(1,2,20,dtype=int)
Eig_vec_test_ad = np.zeros(len(K_list_test))
Eig_vec_test_diff = np.zeros(len(K_list_test))
for idx,ki in enumerate(K_list_test):
    A_ad_test = setup_A(12,ki,t,a,alpha_central,x_left,x_right,advection_v2)
    A_diff_test = setup_A(12,ki,t,a,alpha_central,x_left,x_right,diffusion_v2)
    eigvals_ad_test = np.linalg.eigvals(A_ad_test)
    Eig_vec_test_ad[idx] = np.max(np.abs(eigvals_ad_test))
    eigvals_diff_test = np.linalg.eigvals(A_diff_test)
    Eig_vec_test_diff[idx] = np.max(np.abs(eigvals_diff_test))

plt.figure()
plt.loglog(K_list_test, Eig_vec_test_ad, "--o", label="$\max{|\lambda(\mathcal{A}_{advection})|}$")
# Fit a first-order polynomial (line)
coefficients = np.polyfit(np.log(K_list_test), np.log(Eig_vec_test_ad), 1)
fit = np.poly1d(coefficients)
#plt.loglog(K_list_test, np.exp(fit(np.log(K_list_test))), label="Linear fit")
plt.loglog(K_list_test, K_list_test**(1.0)*25, label="$\mathcal{O}(K^{1.0})$")

plt.loglog(K_list_test, Eig_vec_test_diff, "--o", label="$\max{|\lambda(\mathcal{A}_{diffusion})|}$")
# Fit a first-order polynomial (line)
coefficients = np.polyfit(np.log(K_list_test), np.log(Eig_vec_test_diff), 1)
fit = np.poly1d(coefficients)
#plt.loglog(K_list_test, np.exp(fit(np.log(K_list_test))), label="Quadratic fit")
plt.loglog(K_list_test, (K_list_test**2)*10**(3.5), label="$\mathcal{O}(K^2)$")

# Set specific ticks on the x-axis
x_ticks = [11, 14, 18, 23, 29, 37, 48, 61, 78, 100]
plt.xticks(x_ticks, labels=[str(tick) for tick in x_ticks])  # Add tick labels

plt.title("Scalability Analysis of Largest Eigenvalue ($N=12$)")
plt.legend()
plt.xlabel("K: Number of Elements")
plt.ylabel("$\max{|\lambda(\mathcal{A})|}$")
plt.grid(True, which="both", linestyle="--")  # Add grid for better visualization
plt.show()

# plt.figure()
# plt.imshow(A_ad)
# plt.show()

# plt.figure()
# plt.imshow(A_diff)
# plt.show()

# plt_eig(Eig_mat_ad)
# plt.show()

# plt_eig(Eig_mat_diff)
# plt.show()

# plt_eig_2d(Eig_mat_ad)
# plt.show()

# plt_eig_2d(Eig_mat_diff)
# plt.show()

# eig_scale(Eig_vec_ad,K_list)
# plt.show()

# eig_scale(Eig_vec_diff,K_list,order=3)
# plt.show()

#%%

N_list = np.arange(2,20,2)
K_list = np.arange(2,20,2)

# Create meshgrid
N, K = np.meshgrid(N_list, K_list)

# Define x-ticks for log-log plot
x_ticks = [11, 14, 18, 23, 29, 37, 48, 61, 78, 100]

# Create the subplot
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# 1. Scalability Analysis (Log-Log Plot)
axs[0].loglog(K_list_test, Eig_vec_test_ad, "--o", label="$\max{|\lambda(\mathcal{A}_{advection})|}$")
axs[0].loglog(K_list_test, K_list_test**(1)*22, label="$\mathcal{O}(K^{1})$")
axs[0].loglog(K_list_test, Eig_vec_test_diff, "--o", label="$\max{|\lambda(\mathcal{A}_{diffusion})|}$")
axs[0].loglog(K_list_test, (K_list_test**2)*10**(3.5), label="$\mathcal{O}(K^2)$")
axs[0].set_xticks(x_ticks)
axs[0].set_xticklabels([str(tick) for tick in x_ticks])
axs[0].set_title("Scalability Analysis ($N=12$)")
axs[0].set_xlabel("K: Number of Elements")
axs[0].set_ylabel("$\max{|\lambda(\mathcal{A})|}$")
axs[0].legend()
axs[0].grid(True, which="both", linestyle="--")

# 2. Largest Eigenvalue (Advection) - Pcolormesh Plot
pcolormesh_ad = axs[1].pcolormesh(N, K, np.log10(Eig_mat_ad), shading='auto', cmap="viridis")
cbar_ad = fig.colorbar(pcolormesh_ad, ax=axs[1], orientation='vertical', label="$\log(\max|\lambda(\mathcal{A})|)$")
axs[1].set_title("Largest Eigenvalue (Advection)")
axs[1].set_ylabel("K: number of elements")
axs[1].set_xlabel("N: number of nodes")
axs[1].vlines(12,min(K_list)-1,max(K_list)+1,linestyle="dashed",color="r",label="N=12")
axs[1].legend(loc="upper left")

# 3. Largest Eigenvalue (Diffusion) - Pcolormesh Plot
pcolormesh_diff = axs[2].pcolormesh(N, K, np.log10(Eig_mat_diff), shading='auto', cmap="viridis")
cbar_diff = fig.colorbar(pcolormesh_diff, ax=axs[2], orientation='vertical', label="$\log(\max|\lambda(\mathcal{A})|)$")
axs[2].set_title("Largest Eigenvalue (Diffusion)")
axs[2].set_ylabel("K: number of elements")
axs[2].set_xlabel("N: number of nodes")
axs[2].vlines(12,min(K_list)-1,max(K_list)+1,linestyle="dashed",color="r",label="N=12")
axs[2].legend(loc="upper left")

# Adjust layout
plt.tight_layout()
plt.show()

#%%
N_list = np.arange(2,20,2)
K_list = np.arange(2,20,2)

# Create meshgrid
N, K = np.meshgrid(N_list, K_list)

# Define x-ticks for log-log plot
x_ticks = [11, 14, 18, 23, 29, 37, 48, 61, 78, 100]

# Increase font sizes for the plot
label_fontsize = 20  # Font size for labels (xlabel, ylabel)
legend_fontsize = 14  # Font size for legends
title_fontsize = 22  # Font size for titles
tick_fontsize = 12  # Font size for ticks

# Create the subplot
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# 1. Scalability Analysis (Log-Log Plot)
axs[0].loglog(K_list_test, Eig_vec_test_ad, "--o", label="$\max{|\lambda(\mathcal{A}_{advection})|}$")
axs[0].loglog(K_list_test, K_list_test**(1)*22, label="$\mathcal{O}(K^{1})$")
axs[0].loglog(K_list_test, Eig_vec_test_diff, "--o", label="$\max{|\lambda(\mathcal{A}_{diffusion})|}$")
axs[0].loglog(K_list_test, (K_list_test**2)*10**(3.5), label="$\mathcal{O}(K^2)$")
axs[0].set_xticks(x_ticks)
axs[0].set_xticklabels([str(tick) for tick in x_ticks], fontsize=tick_fontsize)
axs[0].set_title("Scalability Analysis ($N=12$)", fontsize=title_fontsize)
axs[0].set_xlabel("K: Number of Elements", fontsize=label_fontsize)
axs[0].set_ylabel("$\max{|\lambda(\mathcal{A})|}$", fontsize=label_fontsize)
axs[0].legend(fontsize=legend_fontsize)
axs[0].grid(True, which="both", linestyle="--")

# 2. Largest Eigenvalue (Advection) - Pcolormesh Plot
pcolormesh_ad = axs[1].pcolormesh(N, K, np.log10(Eig_mat_ad), shading='auto', cmap="viridis")
cbar_ad = fig.colorbar(pcolormesh_ad, ax=axs[1], orientation='vertical')
cbar_ad.ax.tick_params(labelsize=tick_fontsize)  # Adjust colorbar tick fontsize
cbar_ad.set_label("$\log(\max|\lambda(\mathcal{A})|)$", fontsize=label_fontsize)
axs[1].set_title("Largest Eigenvalue (Advection)", fontsize=title_fontsize)
axs[1].set_ylabel("K: number of elements", fontsize=label_fontsize)
axs[1].set_xlabel("N: number of nodes", fontsize=label_fontsize)
axs[1].vlines(12, min(K_list)-1, max(K_list)+1, linestyle="dashed", color="r", label="N=12")
axs[1].legend(loc="upper left", fontsize=legend_fontsize)

# Adjust layout
plt.tight_layout()
plt.show()

#%%

# Increase font sizes for the plot
label_fontsize = 20  # Font size for labels (xlabel, ylabel)
legend_fontsize = 14  # Font size for legends
title_fontsize = 22  # Font size for titles
tick_fontsize = 10  # Font size for ticks

# Create the subplot
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

N = 12
K_list = np.arange(2, 10, 2)

for K_idx, K in enumerate(K_list):
    # Compute and plot eigenvalues for advection
    A_ad = setup_A(N, K, t, a, alpha_upwind, x_left, x_right, advection_v2)
    eigvals_ad = np.linalg.eigvals(A_ad)
    axs[0].plot(np.real(eigvals_ad), np.imag(eigvals_ad), ".", label=f"$K = {K}$, $N={N}$")

    # Compute and plot eigenvalues for diffusion
    A_diff = setup_A(N, K, t, a, alpha_central, x_left, x_right, diffusion_v2)
    eigvals_diff = np.linalg.eigvals(A_diff)
    axs[1].plot(np.real(eigvals_diff), np.imag(eigvals_diff), ".", label=f"$K = {K}$, $N={N}$")

# Adjust titles, labels, legends, and grids for the first subplot
axs[0].set_title(f"Eigenvalues (Advection)", fontsize=title_fontsize)
axs[0].set_xlabel(r"$\text{Real}(\lambda(\mathcal{A}))$", fontsize=label_fontsize)
axs[0].set_ylabel(r"$\text{Imag}(\lambda(\mathcal{A}))$", fontsize=label_fontsize)
axs[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axs[0].grid(True, which="both", linestyle="--")
axs[0].legend(fontsize=legend_fontsize)

# Adjust titles, labels, legends, and grids for the second subplot
axs[1].set_title(f"Eigenvalues (Diffusion)", fontsize=title_fontsize)
axs[1].set_xlabel(r"$\text{Real}(\lambda(\mathcal{A}))$", fontsize=label_fontsize)
axs[1].set_ylabel(r"$\text{Imag}(\lambda(\mathcal{A}))$", fontsize=label_fontsize)
axs[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axs[1].grid(True, which="both", linestyle="--")
axs[1].legend(fontsize=legend_fontsize)

# Adjust layout
plt.tight_layout()
plt.show()


#%%









