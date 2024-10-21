import numpy as np
import sys
sys.path.insert(0,r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\2_semester\Advanced nummerical\AdvSciComp\Assignment2\func")
from fourier import nodes, diff_matrix
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import norm
from L2space import discrete_inner_product

def compute_L2_error(numerical, exact, weights):
    return np.sqrt(discrete_inner_product(numerical - exact, numerical - exact, weights))


#%% Opgave c)

N = 50
x = nodes(N)

x1 = 20
x2 = 20

D = diff_matrix(N)
D3 = D@D@D
a = 2*np.pi/(x1+x2)

def f(t,u,D,D3):
    return -6*a*u*D@u - a**3 * D3@u

x0 = np.pi

def u_exact(w,t,c,w0):
    x  = w*(x1+x2)/(2*np.pi) - x1
    x0 = w0*(x1+x2)/(2*np.pi) - x1
    return 0.5*c*1/(np.cosh(0.5*np.sqrt(c)*(x-c*t-x0)))**2

#%% Convergence test:
# Set up different values for N (number of grid points)
N_values = [10,20,25,30,35,40,45,50]
errors = []

# Constants
c_value = 0.25
tf = 1.0
alpha = 0.5
x1, x2 = 20, 20
x0 = np.pi

for N in N_values:
    # Grid points and differentiation matrices
    x = nodes(N)
    D = diff_matrix(N)
    D3 = D @ D @ D
    a = 2 * np.pi / (x1 + x2)

    # Initial condition
    u0 = u_exact(x, 0, c_value, x0)

    # Time integration
    max_step = alpha * 1.73 * 8 / (N**3 * a**3)
    sol = solve_ivp(f, [0, tf], u0, args=(D, D3), max_step=max_step,dense_output=True, method="RK23")
    
    # Extract solution at final time
    U_approx = sol.y[:, -1]

    # Exact solution at final time
    U_exact = u_exact(x, tf, c_value, x0)

    # Compute the L2 error
    error = np.max(np.abs(U_approx-U_exact))
    errors.append(error)

plt.figure()
plt.loglog(N_values,errors)


#%% Opgave d)
N = 10
c = np.array([0.25,0.5,1])

alpha = 0.9
max_step = alpha * 1.73*8/(N**3*a**3)

for c_i in c:
    u0 = u_exact(x,0,c_i,x0)
    tf = 1.0
    sol = solve_ivp(f,[0, tf],u0,args=(D,D3),max_step=max_step,dense_output=True,method="RK23")

    t = sol.t
    U = sol.y.T
    plt.figure()
    X,T = np.meshgrid(x,t)

    plt.pcolormesh(T,X,U)
    plt.xlabel("t: time")
    plt.ylabel("x: space")
    plt.title("Diffusion Equation")

    plt.figure()
    plt.plot(x,U[0],".-",label=r"$u(x,0) (Approx)$")
    plt.plot(x,U[-1],".-",label=r"$u(x,T)$ (Approx)")
    plt.plot(x,u_exact(x,tf,c_i,x0),"--",label=r"$u(x,T)$ (Exact)")
    plt.legend()

    int_M_test = np.zeros(len(t))
    int_V_test = np.zeros(len(t))
    int_E_test = np.zeros(len(t))
    L2_error = np.zeros(len(t))   # L2-norm error over time
    Linf_error = np.zeros(len(t)) # Linf-norm error over time
    
    for i in range(len(t)):
        w = np.ones_like(x)*2*np.pi/N
        int_M_test[i] = discrete_inner_product(np.ones_like(U[i]),U[i],w)
        int_V_test[i] = discrete_inner_product(U[i],U[i],w)
        int_E_term1   = discrete_inner_product(D@U[i],D@U[i],w)
        int_E_term2   = discrete_inner_product(U[i],U[i]*U[i],w)
        int_E_test[i] = 0.5*int_E_term1 - int_E_term2

        # Numerical solution at time t[i]
        U_approx = U[i]
        
        # Exact solution at time t[i]
        U_exact = u_exact(x, t[i], c_i, x0)
        
        # Compute L2 norm of the error
        L2_error[i] = np.sqrt(discrete_inner_product(U_approx - U_exact, U_approx - U_exact, w))
        
        # Compute Linf norm of the error (max absolute error)
        Linf_error[i] = np.max(np.abs(U_approx - U_exact))
    
    plt.figure()
    plt.plot(t,int_M_test,label=r"$\int u(x,t) dx$")
    plt.plot(t,int_V_test,label=r"$\int u(x,t)^2 dx$")
    plt.plot(t,int_E_test,label=r"$\int \frac{1}{2}u_x(x,t)^2 - u^3 dx$")
    plt.xlabel("t")
    plt.legend()

    # Plot the L2 norm error evolution over time
    plt.figure()
    plt.plot(t, L2_error, label=r"$\|u - \mathcal{I}_N u\|_{L_2}$")
    plt.xlabel('Time (t)')
    plt.ylabel(r"$L_2$-norm Error")
    plt.title(r'Evolution of $L_2$ Error')
    plt.legend()
    plt.grid(True)

    # Plot the Linf norm error evolution over time
    plt.figure()
    plt.plot(t, Linf_error, label=r"$\|u - \mathcal{I}_N u\|_{L_\infty}$", color="red")
    plt.xlabel('Time (t)')
    plt.ylabel(r"$L_\infty$-norm Error")
    plt.title(r'Evolution of $L_\infty$ Error')
    plt.legend()
    plt.grid(True)


plt.show()




