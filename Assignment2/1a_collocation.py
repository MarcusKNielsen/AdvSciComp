import numpy as np 
import matplotlib.pyplot as plt
from JacobiGL import JacobiGL

def discrete_inner_product(u,v,w):    
    return np.sum(u*v*w)

def discrete_L2_norm(u,w):
    return np.sqrt(discrete_inner_product(u,u,w))

def JacobiP(x,alpha,beta,N):

    if x.size > 1:
        J_nm2 = np.ones(len(x)) # J_{n-2}
    else:
        J_nm2 = 1
    J_nm1 = 1/2*(alpha-beta+(alpha+beta+2)*x) # J_{n-1}

    if N==0:
        return J_nm2
    elif N==1:
        return J_nm1

    for n in range(1,N):

        # Computing the recursive coefficients
        anm2  = 2*(n+alpha)*(n+beta)/( (2*n+alpha+beta+1)*(2*n+alpha+beta) )
        anm1  = (alpha**2-beta**2)/( (2*n+alpha+beta+2)*(2*n+alpha+beta) )
        an    = 2*(n+1)*(n+beta+alpha+1)/( (2*n+alpha+beta+2)*(2*n+alpha+beta+1) )

        # Computing
        J_n = ( (anm1 + x )*J_nm1 - anm2*J_nm2 ) / an

        # Updating step
        J_nm2 = J_nm1
        J_nm1 = J_n
    
    return J_n

def get_vandermonde(x_GL):
    N = len(x_GL)
    V = np.zeros([N,N])
    for j in range(N):
        V[:,j] = JacobiP(x_GL,alpha=0,beta=0,N=j)
    return V

def GradJacobiP(x,alpha,beta,N):

    if N == 0:
        return 0

    return 1/2*(alpha+beta+N+1)*JacobiP(x,alpha+1,beta+1,N-1)

def get_vandermonde_x(x_GL):
    N = len(x_GL)
    V = np.zeros([N,N])
    for j in range(N):
        V[:,j] = GradJacobiP(x_GL,alpha=0,beta=0,N=j)
    return V

def D_poly(x_GL):
    Vx = get_vandermonde_x(x_GL)
    V = get_vandermonde(x_GL)

    return Vx@np.linalg.inv(V)

def Ln_mat(eps,D):

    Ln = -4*eps*D@D-2*D

    return Ln

def u_exact(x,epsilon):
    return (np.exp(-x / epsilon) + (x - 1) - np.exp(-1 / epsilon )*x) / (np.exp(-1 / epsilon) - 1)

def solve_Collocation(N,epsilon_values):

    # Magic points
    x_GL = JacobiGL(0,0,N)

    # Right hand side
    f = np.ones_like(x_GL)

    # Initialization of solution
    u_solutions = np.zeros([len(epsilon_values),len(x_GL)])

    for idx,eps in enumerate(epsilon_values):

        D = D_poly(x_GL)
        Ln = Ln_mat(eps,D)

        # Adding boundary conditions
        Ln[0] = 0
        Ln[0,0] = 1
        Ln[-1] = 0
        Ln[-1,-1] = 1

        f[0] = 0
        f[-1] = 0

        u_solutions[idx] = np.linalg.solve(Ln,f)

    return u_solutions



#%%

N = 32
epsilon_values = np.array([0.1, 0.01, 0.001])

u_solutions = solve_Collocation(N, epsilon_values)
x_GL = JacobiGL(0, 0, N)

# Prepare the figure with a 2x2 grid of subplots
fig, ax = plt.subplots(2, 2, figsize=(8, 8))
x_lin = np.linspace(-1, 1, 1000)

# Plot the solutions for each epsilon on the first three subplots
for idx, eps in enumerate(epsilon_values):
    row, col = divmod(idx, 2)
    ax[row, col].plot(x_lin, u_exact((x_lin+1)/2, eps), label="Exact")
    ax[row, col].plot(x_GL, u_solutions[idx], ".", label=rf"Numerical")

    # Customize the subplot
    ax[row, col].set_title(rf"Solution for $\varepsilon$={eps} with N={N}")
    ax[row, col].legend()
    ax[row, col].set_xlabel('x')
    ax[row, col].set_ylabel('u(x)')

# Convergence plot in the last subplot
#N_array = np.arange(10, 150, 5)
N_array = np.logspace(np.log10(4), np.log10(200), num=50, dtype=int)
N_array = np.unique(N_array)

error_temp = np.zeros([len(N_array), len(epsilon_values)])

for N_idx, N in enumerate(N_array):
    x_GL = JacobiGL(0, 0, N)
    
    u_exact_mat_temp = np.zeros([len(epsilon_values), len(x_GL)])
    u_exact_mat_temp[0] = u_exact((x_GL+1)/2, epsilon_values[0]) 
    u_exact_mat_temp[1] = u_exact((x_GL+1)/2, epsilon_values[1]) 
    u_exact_mat_temp[2] = u_exact((x_GL+1)/2, epsilon_values[2]) 

    u_solutions_temp = solve_Collocation(N, epsilon_values)
    
    error_temp[N_idx] = np.max(np.abs(u_exact_mat_temp - u_solutions_temp), axis=1)

# Plot the convergence in the last subplot (ax[1,1])
ax[1, 1].set_title("Convergence Plot")
for i in range(3):
    ax[1, 1].semilogy(N_array, error_temp[:, i], ".-", label=rf"$\epsilon$={epsilon_values[i]}")
ax[1, 1].legend()
ax[1, 1].set_xlabel('N')
ax[1, 1].set_ylabel(r'$\Vert u - u_N \Vert$')

# Adjust layout to prevent overlap
plt.tight_layout(pad=2.0, w_pad=1.0, h_pad=2.5)

# Show the plots
plt.show()

