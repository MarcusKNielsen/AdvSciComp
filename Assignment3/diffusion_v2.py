import pandas as pd
import numpy as np
import func.L2space as L2space
import func.legendre as legendre
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy

def flux_star_right(u,up1,alpha,a):
        flux = a*(u[-1]+up1[0])/2 + np.abs(a)*(1-alpha)/2*(u[-1]-up1[0])
        return flux

def flux_star_left(u,um1,alpha,a):
    flux = a*(u[0]+um1[-1])/2 + np.abs(a)*(1-alpha)/2*(-u[0]+um1[-1])
    return flux 

def flux_star(um,up,alpha,a):
        flux = a*(up+um)/2 + np.abs(a)*(1-alpha)/2*(up-um)
        return flux

def f_func(t,u,Mk_inv,D,S,N,alpha,a):
    
    DUDT = np.zeros_like(u)

    q = np.zeros_like(u)
    for k in range(0,len(u),N):
         q[k:int(k+N)] = D@u[k:int(k+N)]

    for k in range(0,len(u),N):

        uk = u[k:int(k+N)]
        qk = q[k:int(k+N)]

        lagrange_rhs_left = np.zeros(N)
        lagrange_rhs_left[0] = 1
        lagrange_rhs_right = np.zeros(N)
        lagrange_rhs_right[-1] = 1

        if k == 0:

            # Flux for Uk equation
            # Flux left
            qm_left      = q[k] 
            flux_left_U  = -a*qm_left 
            # Flux right
            qm_right   = q[k+N-1]
            qp_right   = q[k+N] 
            flux_right_U = flux_star(qp_right,qm_right,alpha,a)

            # Flux for Q equation
            # Flux left
            um_left     = u[k] 
            flux_left_Q  = a*um_left 
            # Flux right
            um_right   = u[k+N-1]
            up_right   = u[k+N] 
            flux_right_Q = flux_star(up_right,um_right,alpha,a)

            rhs_u = lagrange_rhs_right*(np.sqrt(a)*qm_right-flux_right_U)-lagrange_rhs_left*(np.sqrt(a)*qm_left-flux_left_U)
            rhs_q = lagrange_rhs_right*(np.sqrt(a)*um_right-flux_right_Q)-lagrange_rhs_left*(np.sqrt(a)*um_left-flux_left_Q)

        elif k == (len(u)-N):

            # Flux for Uk equation
            # Flux left
            qm_left   = q[k]
            qp_left   = q[k-1] 
            flux_left_U = flux_star(qm_left,qp_left,alpha,a)

            # Flux right
            qm_right      = q[-1] 
            flux_right_U  = -a*qm_right

            # Flux for Q equation
            # Flux left
            um_left   = u[k]
            up_left   = u[k-1] 
            flux_left_Q = flux_star(um_left,up_left,alpha,a)

            # Flux right
            um_right      = u[-1] 
            flux_right_Q  = a*um_right

            rhs_u = lagrange_rhs_right*(np.sqrt(a)*qm_right-flux_right_U)-lagrange_rhs_left*(np.sqrt(a)*qm_left-flux_left_U)
            rhs_q = lagrange_rhs_right*(np.sqrt(a)*um_right-flux_right_Q)-lagrange_rhs_left*(np.sqrt(a)*um_left-flux_left_Q)

        else:

            # Flux for Uk equation
            # left boundary of element
            qm_left    = q[k]
            qp_left    = q[k-1] 
            flux_left_U  = flux_star(qm_left,qp_left,alpha,a)
            
            # right boundary of element
            qm_right   = q[k+N-1]
            qp_right   = q[k+N]
            flux_right_U = flux_star(qp_right,qm_right,alpha,a)

            # Flux for Q equation
            # left boundary of element
            um_left    = u[k]
            up_left    = u[k-1] 
            flux_left_Q  = flux_star(um_left,up_left,alpha,a)
            
            # right boundary of element
            um_right   = u[k+N-1]
            up_right   = u[k+N]
            flux_right_Q = flux_star(up_right,um_right,alpha,a)

            rhs_u = lagrange_rhs_right*(np.sqrt(a)*qm_right-flux_right_U)-lagrange_rhs_left*(np.sqrt(a)*qm_left-flux_left_U)
            rhs_q = lagrange_rhs_right*(np.sqrt(a)*um_right-flux_right_Q)-lagrange_rhs_left*(np.sqrt(a)*um_left-flux_left_Q)

        qk = Mk_inv@(S@(np.sqrt(a)*uk)-rhs_q)
        DUDT[k:int(k+N)] = Mk_inv@(S@(np.sqrt(a)*qk)-rhs_u) 

    return DUDT 

def total_grid_points(number_element,x_nodes,a,b):

    k_boundary = np.linspace(a,b,(number_element+1))
    x_total = np.zeros(len(x_nodes)*number_element)
    x_total = np.array([(x_nodes+1)/2*(xr-xl)+xl for xl,xr in zip(k_boundary[:-1],k_boundary[1:])])

    return x_total.ravel()

def u_exact(x,t,a):
     return np.exp(-x**2/(4*a*t))/np.sqrt(4*np.pi*a*t)

if __name__ == "__main__":
     
    x_left = -1
    x_right = 1
    N = 4
    number_element = 30
    x_nodes = legendre.nodes(N)
    x_total = total_grid_points(number_element,x_nodes,x_left,x_right)

    h = (x_total[-1]-x_total[0])/number_element

    V,Vx,_ = legendre.vander(x_nodes)
    M = np.linalg.inv(V@V.T)
    Mk = (h/2)*M
    Mk_inv = np.linalg.inv(Mk) 
    Dx = Vx@np.linalg.inv(V)
    S = M@Dx
    a = 1
    alpha = 1 
    max_step = 0.1
    t0 = 0.005
    tf = 0.009

    u0 = u_exact(x_total,t0,a)

    sol = solve_ivp(f_func, [t0, tf], u0, args=(Mk_inv,Dx,S,N,alpha,a), max_step=max_step, dense_output=True, method="Radau")

    plt.figure()
    plt.plot(x_total,sol.y[:,-1],'.',label=r"$u(x,t_f)$")
    plt.plot(x_total,sol.y[:,0],'-',label=r"$u(x,t_0)$")
    plt.plot(x_total,u_exact(x_total,tf,a),label=r"$u_{exact}(x,t_f)$")
    plt.legend()

    print(np.max(np.abs(u_exact(x_total,tf,a) - sol.y[:,-1]))) 

    plt.figure()
    X, T = np.meshgrid(x_total, sol.t)

    # Create the pcolormesh plot
    pcm = plt.pcolormesh(T, X, sol.y.T)

    # Label the axes and add a title
    plt.xlabel("t: time")
    plt.ylabel("x: space")
    plt.title("Diffusion Equation")

    # Add the colorbar
    plt.colorbar(pcm, label="u(x,t)")

    plt.show()
