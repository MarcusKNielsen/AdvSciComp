import numpy as np
from fourier import nodes, diff_matrix
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import norm
from L2space import discrete_inner_product

N = 52
x = nodes(N)


x1 = 10
x2 = 10

D = diff_matrix(N)
D3 = D@D@D
def f(t,u,D,D3):
    a = 2*np.pi/(x1+x2)
    return -6*a*u*D@u - a**3 * D3@u

c  = 0.25
x0 = np.pi

def u_exact(w,t,c,w0):
    x  = w*(x1+x2)/(2*np.pi) - x1
    x0 = w0*(x1+x2)/(2*np.pi) - x1
    return 0.5*c*1/(np.cosh(0.5*np.sqrt(c)*(x-c*t-x0)))**2

c = np.array([0.25,0.5,1])

alpha = 0.9
max_step = alpha * 1.73*8/N**3

for c_i in c:
    u0 = u_exact(x,0,c_i,x0)
    T = 1.0
    sol = solve_ivp(f,[0, T],u0,args=(D,D3),max_step=max_step,dense_output=True,method="RK23")

    t = sol.t
    U = sol.y.T
    plt.figure()
    X,T = np.meshgrid(x,t)

    plt.pcolormesh(T,X,U)
    plt.xlabel("t: time")
    plt.ylabel("x: space")
    plt.title("Diffusion Equation")
    plt.show()


    plt.figure()
    plt.plot(x,U[0],".-",label=r"$u(x,0)$")
    plt.plot(x,U[-1],".-",label=r"u(x,T)")
    plt.legend()
    plt.show()


    int_M_test = np.zeros(len(t))
    int_V_test = np.zeros(len(t))
    int_E_test = np.zeros(len(t))
    
    for i in range(len(t)):
        w = np.ones_like(x)*2*np.pi/N
        int_M_test[i] = discrete_inner_product(np.ones_like(U[i]),U[i],w)
        int_V_test[i] = discrete_inner_product(U[i],U[i],w)
        int_E_term1   = discrete_inner_product(D@U[i],D@U[i],w)
        int_E_term2   = discrete_inner_product(U[i],U[i]*U[i],w)
        int_E_test[i] = 0.5*int_E_term1 - int_E_term2
    
    plt.figure()
    plt.plot(t,int_M_test,label=r"$\int u(x,t) dx$")
    plt.plot(t,int_V_test,label=r"$\int u(x,t)^2 dx$")
    plt.plot(t,int_E_test,label=r"$\int \frac{1}{2}u_x(x,t)^2 - u^3 dx$")
    plt.xlabel("t")
    plt.legend()
    plt.show()


