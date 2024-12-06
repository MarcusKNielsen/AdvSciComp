#%% modules
import pandas as pd
import numpy as np
import func.L2space as L2space
import func.legendre as legendre
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy
import advection_v2 
import diffusion_v2


def scalability(method,order=1):
    #% Checking only for 100 elements and N
    K_list_test = np.arange(2,110,10)
    Eig_vec_test = np.zeros(len(K_list_test))
    for idx,ki in enumerate(K_list_test):
        A_ad_test = setup_A(12,ki,t,a,0,x_left,x_right,method)
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
N_list = np.arange(2,10,2)
K_list = np.arange(2,10,2)

Eig_mat_ad = np.zeros([len(N_list),len(K_list)])
Eig_vec_ad = np.zeros(len(N_list))
Eig_mat_diff = np.zeros([len(N_list),len(K_list)])
Eig_vec_diff = np.zeros(len(N_list))

for N_idx,N in enumerate(N_list):
    for K_idx,K in enumerate(K_list):

        A_ad = setup_A(N,K,t,a,0,x_left,x_right,advection_v2)
        eigvals_ad = np.linalg.eigvals(A_ad)
        Eig_mat_ad[N_idx,K_idx] = np.max(np.abs(eigvals_ad))

        if N == K:
            Eig_vec_ad[N_idx] = np.max(np.abs(eigvals_ad))

        A_diff = setup_A(N,K,t,a,1,x_left,x_right,diffusion_v2)
        eigvals_diff = np.linalg.eigvals(A_diff)
        Eig_mat_diff[N_idx,K_idx] = np.max(np.abs(eigvals_diff))

        if N == K:
            Eig_vec_diff[N_idx] = np.max(np.abs(eigvals_diff))


scalability(advection_v2)
scalability(diffusion_v2,order=2)


#% Checking only for 100 elements and N
K_list_test = np.logspace(1,1.5,20,dtype=int)
Eig_vec_test_ad = np.zeros(len(K_list_test))
Eig_vec_test_diff = np.zeros(len(K_list_test))
for idx,ki in enumerate(K_list_test):
    A_ad_test = setup_A(12,ki,t,a,0,x_left,x_right,advection_v2)
    A_diff_test = setup_A(12,ki,t,a,0,x_left,x_right,diffusion_v2)
    eigvals_ad_test = np.linalg.eigvals(A_ad_test)
    Eig_vec_test_ad[idx] = np.max(np.abs(eigvals_ad_test))
    eigvals_diff_test = np.linalg.eigvals(A_diff_test)
    Eig_vec_test_diff[idx] = np.max(np.abs(eigvals_diff_test))

plt.figure()
plt.loglog(K_list_test,Eig_vec_test_ad,"--o",label="$\max{|\lambda(\mathcal{A}_{advection})|}$")
# Fit a first-order polynomial (line)
coefficients = np.polyfit(K_list_test, Eig_vec_test_ad, 1)
fit = np.poly1d(coefficients)
#plt.loglog(K_list_test,fit(K_list_test),label="Linear fit")
plt.loglog(K_list_test,K_list_test**(1.2)*20,label="$\mathcal{O}(K^{1.2})$")

plt.loglog(K_list_test,Eig_vec_test_diff,"--o",label="$\max{|\lambda(\mathcal{A}_{diffusion})|}$")
# Fit a first-order polynomial (line)
coefficients = np.polyfit(K_list_test, Eig_vec_test_diff, 1)
fit = np.poly1d(coefficients)
#plt.loglog(K_list_test,fit(K_list_test),label="Quadradic fit")
plt.loglog(K_list_test,(K_list_test**2)*10**(3.5),label="$\mathcal{O}(K^2)$")

plt.title("Scalability analysis with variying $K$ and fixed $N=12$")
plt.legend()
plt.xlabel("K")
plt.ylabel("$\max{|\lambda(\mathcal{A})|}$")

plt.figure()
plt.imshow(A_ad)
plt.figure()
plt.imshow(A_diff)

plt_eig(Eig_mat_ad)
plt_eig(Eig_mat_diff)

plt_eig_2d(Eig_mat_ad)
plt_eig_2d(Eig_mat_diff)

eig_scale(Eig_vec_ad,K_list)
eig_scale(Eig_vec_diff,K_list,order=3)


plt.figure()

plt.scatter(np.real(eigvals_ad),np.imag(eigvals_ad))


plt.figure()

plt.scatter(np.real(eigvals_diff),np.imag(eigvals_diff))


plt.show()
