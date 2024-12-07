import pandas as pd
import numpy as np
import func.L2space as L2space
import func.legendre as legendre
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy

def flux_star(um,up,alpha,a):
        flux = a*(up+um)/2 + np.abs(a)*(1-alpha)/2*(up-um)
        return flux

def f_func(t,u,Mk_inv,D,S,N,alpha,a,d,formulation):
    
    DUDT = np.zeros_like(u)

    lagrange_rhs_left = np.zeros(N)
    lagrange_rhs_left[0] = 1
    lagrange_rhs_right = np.zeros(N)
    lagrange_rhs_right[-1] = 1

    q = (D @ (u.reshape(-1, N)).T).T.ravel()  # Apply D to each block and reshape back

    for k in range(0,len(u),N):

        uk = u[k:int(k+N)]
        qk = q[k:int(k+N)] 

        if k == 0:

            #%% Diffusion

            # Flux for Uk equation
            # Flux left
            qm_left   = q[k]
            qp_left   = -qm_left
            flux_left_U = flux_star(qm_left,qp_left,alpha,np.sqrt(d))
 
            # Flux right
            qm_right   = q[k+N-1]
            qp_right   = q[k+N] 
            flux_right_U = flux_star(qp_right,qm_right,alpha,np.sqrt(d))

            # Flux for Q equation
            # Flux left
            um_left     = u[k] 
            flux_left_Q  = np.sqrt(d)*um_left 
            # Flux right
            um_right   = u[k+N-1]
            up_right   = u[k+N] 
            flux_right_Q = flux_star(up_right,um_right,alpha,np.sqrt(d))

            #%% Advection
            # left boundary of element 
            #up_left    = 0 
            #um_left    = u[k] 
            #flux_left_ad  = a*up_left 
            
            um_left   = u[k]
            up_left   = -um_left
            flux_left_ad = flux_star(um_left,up_left,alpha,a)

            # right boundary of element 
            um_right   = u[k+N-1]
            up_right   = u[k+N] 
            flux_right_ad = flux_star(up_right,um_right,alpha,a)

        elif k == (len(u)-N):

            #%% Diffusion
            # Flux for Uk equation 
            # Flux left
            qm_left   = q[k]
            qp_left   = q[k-1] 
            flux_left_U = flux_star(qm_left,qp_left,alpha,np.sqrt(d))

            # Flux right
            qm_right      = q[-1] 
            qp_right      = -qm_right
            flux_right_U  = flux_star(qp_right,qm_right,alpha,np.sqrt(d))

            # Flux for Q equation 
            # Flux left
            um_left   = u[k]
            up_left   = u[k-1] 
            flux_left_Q = flux_star(um_left,up_left,alpha,np.sqrt(d))

            # Flux right
            um_right      = u[-1] 
            flux_right_Q  = np.sqrt(d)*um_right

            #%% Advection
            # left boundary of element 
            um_left    = u[k]
            up_left    = u[k-1] 
            flux_left_ad  = flux_star(um_left,up_left,alpha,a)
            
            # right boundary of element  
            #um_right   = u[-1] 
            #flux_right_ad = a*um_right 
            um_right      = u[-1] 
            up_right      = -um_right
            flux_right_ad  = flux_star(up_right,um_right,alpha,a)

        else:
            
            #%% Diffusion

            # Flux for Uk equation
            # left boundary of element 
            qm_left    = q[k]
            qp_left    = q[k-1] 
            flux_left_U  = flux_star(qm_left,qp_left,alpha,np.sqrt(d))
            
            # right boundary of element 
            qm_right   = q[k+N-1]
            qp_right   = q[k+N]
            flux_right_U = flux_star(qp_right,qm_right,alpha,np.sqrt(d))

            # Flux for Q equation
            # left boundary of element
            um_left    = u[k]
            up_left    = u[k-1] 
            flux_left_Q  = flux_star(um_left,up_left,alpha,np.sqrt(d))
            
            # right boundary of element
            um_right   = u[k+N-1]
            up_right   = u[k+N]
            flux_right_Q = flux_star(up_right,um_right,alpha,np.sqrt(d))

            #%% Advection
            # left boundary of element
            um_left    = u[k]
            up_left    = u[k-1] 
            flux_left_ad  = flux_star(um_left,up_left,alpha,a)
            
            # right boundary of element 
            um_right   = u[k+N-1]
            up_right   = u[k+N]
            flux_right_ad = flux_star(up_right,um_right,alpha,a)

        if formulation == "w":
            
            rhs_u = lagrange_rhs_right*(flux_right_U)-lagrange_rhs_left*(flux_left_U)
            rhs_q = lagrange_rhs_right*(flux_right_Q)-lagrange_rhs_left*(flux_left_Q)
            rhs_ad = lagrange_rhs_right*(flux_right_ad)-lagrange_rhs_left*(flux_left_ad)
            
            qk = Mk_inv@(-((S.T)@(np.sqrt(d)*uk))+rhs_q)
            DUDT[k:int(k+N)] = Mk_inv@(-((S.T)@(np.sqrt(d)*qk))+rhs_u) + Mk_inv@(((S.T)@(a*uk))-rhs_ad) 

        elif formulation == "s":
            
            rhs_u = lagrange_rhs_right*(np.sqrt(d)*qm_right-flux_right_U)-lagrange_rhs_left*(np.sqrt(d)*qm_left-flux_left_U)
            rhs_q = lagrange_rhs_right*(np.sqrt(d)*um_right-flux_right_Q)-lagrange_rhs_left*(np.sqrt(d)*um_left-flux_left_Q)
            rhs_ad = lagrange_rhs_right*(a*um_right-flux_right_ad)-lagrange_rhs_left*(a*um_left-flux_left_ad)

            qk = Mk_inv@(S@(np.sqrt(d)*uk)-rhs_q) 
            DUDT[k:int(k+N)] = Mk_inv@(S@(np.sqrt(d)*qk)-rhs_u-S@(a*uk)+rhs_ad)  

    return DUDT 

def total_grid_points(number_element,x_nodes,a,b):

    k_boundary = np.linspace(a,b,(number_element+1))
    x_total = np.zeros(len(x_nodes)*number_element)
    x_total = np.array([(x_nodes+1)/2*(xr-xl)+xl for xl,xr in zip(k_boundary[:-1],k_boundary[1:])])

    return x_total.ravel()

def u_exact(x,t,a,d):
     return np.exp(-(x-a*t)**2/(4*d*t))/np.sqrt(4*np.pi*d*t)

if __name__ == "__main__":
    x_left = -5
    x_right = 5
    N = 10
    number_element = 10
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
    d = 1.2
    alpha = 1
    max_step = 0.001
    t0 = 0.08
    tf = 1.5
    formulation = "s"

    u0 = u_exact(x_total,t0,a,d)

    sol = solve_ivp(f_func, [t0, tf], u0, args=(Mk_inv,Dx,S,N,alpha,a,d,formulation), max_step=max_step, dense_output=True, method="Radau")

    plt.figure()
    plt.plot(x_total,sol.y[:,-1],'o',label=r"$u(x,t_f)$")
    plt.plot(x_total,sol.y[:,0],'-',label=r"$u(x,t_0)$")
    plt.plot(x_total,u_exact(x_total,tf,a,d),label=r"$u_{exact}(x,t_f)$")
    plt.legend()

    plt.figure()
    X, T = np.meshgrid(x_total, sol.t)

    # Create the pcolormesh plot
    pcm = plt.pcolormesh(T, X, sol.y.T)

    # Label the axes and add a title
    plt.xlabel("t: time")
    plt.ylabel("x: space")
    plt.title("Collision of Two Solitons")

    # Add the colorbar
    plt.colorbar(pcm, label="u(x,t)")

    I = np.eye(number_element)
    M_total = np.kron(I,Mk)
    
    integral = np.sum(M_total@sol.y,axis=0)
    print(integral)

    plt.show()
