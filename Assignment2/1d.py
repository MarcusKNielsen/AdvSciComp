#%% Importing modules
import numpy as np
import sys
sys.path.insert(0,r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\2_semester\Advanced nummerical\AdvSciComp\Assignment2\func")
from fourier import nodes, diff_matrix
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from L2space import discrete_inner_product
#import Assignment2.functions_TIME as functions
import functions_TIME as functions

dealias = False

#%% Opgave d)
N = 400
# Fine grid (zero-padding)
M = 3*N//2
w = nodes(N)
D = diff_matrix(N)
D3 = D @ D @ D
x1, x2 = 40, 40
a = 2 * np.pi / (x1 + x2)
w0 = np.pi 
tf = 1.0

c = np.array([0.25,0.5,1])

alpha = 0.5
max_step = alpha * 1.73*8/(N**3*a**3)

time = True

for c_i in c:

    if dealias:
        x = w*(x1+x2)/(2*np.pi) - x1 
        x0 = w0*(x1+x2)/(2*np.pi) - x1
        u0 = functions.dealias_IC(N,M,w0,x1,x2,c_i) 
    else:
        x = w*(x1+x2)/(2*np.pi) - x1 
        x0 = w0*(x1+x2)/(2*np.pi) - x1
        u0 = functions.u_exact(x, 0, c_i, x0)

    if dealias:
        sol = solve_ivp(functions.f_alias_free,[0, tf],u0,args=(D,D3,a,N,M),max_step=max_step,dense_output=True,method="RK23")
    else:
        sol = solve_ivp(functions.f, [0, tf], u0, args=(D, D3,a), max_step=max_step,dense_output=True, method="RK23")

    t = sol.t
    U = sol.y.T
    plt.figure()
    X,T = np.meshgrid(x,t)

    plt.pcolormesh(T,X,U)
    plt.xlabel("t: time")
    plt.ylabel("x: space")
    plt.title(f"Diffusion Equation: c={c_i}")

    plt.figure()
    plt.plot(x,U[0],".-",label=r"$u(x,0) (Approx)$")
    plt.plot(x,U[-1],".-",label=r"$u(x,T)$ (Approx)")
    plt.plot(x,functions.u_exact(x,tf,c_i,x0),"--",label=r"$u(x,T)$ (Exact)")
    plt.plot(x,functions.u_exact(x,0,c_i,x0),"--",label=r"$u(x,0)$ (Exact)")
    plt.title(f"c={c_i}")
    plt.legend()

    if time:

        int_M_test = np.zeros(len(t))
        int_V_test = np.zeros(len(t))
        int_E_test = np.zeros(len(t))
        L2_error = np.zeros(len(t))   # L2-norm error over time
        Linf_error = np.zeros(len(t)) # Linf-norm error over time
        
        for i in range(len(t)):
            weight = np.ones_like(x)*2*np.pi/N
            int_M_test[i] = discrete_inner_product(np.ones_like(U[i]),U[i],weight)
            int_V_test[i] = discrete_inner_product(U[i],U[i],weight)
            int_E_term1   = discrete_inner_product(D@U[i],D@U[i],weight)
            int_E_term2   = discrete_inner_product(U[i],U[i]*U[i],weight)
            int_E_test[i] = 0.5*int_E_term1 - int_E_term2

            # Numerical solution at time t[i]
            U_approx = U[i]
            
            # Exact solution at time t[i]
            x  = w*(x1+x2)/(2*np.pi) - x1
            x0 = w0*(x1+x2)/(2*np.pi) - x1
            U_exact = functions.u_exact(x, t[i], c_i, x0)
            
            # Compute L2 norm of the error
            L2_error[i] = np.sqrt(discrete_inner_product(U_approx - U_exact, U_approx - U_exact, weight))
            
            # Compute Linf norm of the error (max absolute error)
            Linf_error[i] = np.max(np.abs(U_approx - U_exact))


        plt.figure()
        plt.plot(t,int_M_test,label=r"M= $\int u(x,t) dx$")
        plt.plot(t,int_V_test,label=r"V= $\int u(x,t)^2 dx$")
        plt.plot(t,int_E_test,label=r"E= $\int \frac{1}{2}u_x(x,t)^2 - u^3 dx$")
        plt.title(f"Mass (M), Momentum (V), Energy (E), for c={c_i}")
        plt.xlabel("t")
        plt.legend()
    
        # Plot the L2 norm error evolution over time
        plt.figure()
        plt.plot(t, L2_error, label=r"$\|u - \mathcal{I}_N u\|_{L_2}$")
        plt.xlabel('Time (t)')
        plt.ylabel(r"Error")
        plt.title(f'Evolution Errors for c={c_i}')
        plt.plot(t, Linf_error, label=r"$\|u - \mathcal{I}_N u\|_{L_\infty}$", color="red")
        plt.legend()
        plt.grid(True)

plt.show()
    