import numpy as np
#import func.L2space as L2space
import func.legendre as legendre
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
#import scipy


def flux_star(um,up,alpha,a):
        flux = a*(up+um)/2 + np.abs(a)*(1-alpha)/2*(up-um)
        return flux

def f_func(t,u,Mk_inv,D,S,N,alpha,a,formulation):
    
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

            # Flux for Uk equation
            
            # Flux left (Reflective) 
            qm_left   = q[k]
            qp_left   = -qm_left
            flux_left_U = flux_star(qm_left,qp_left,alpha,np.sqrt(a))
            
            # left boundary of element (Periodic)
            # qp_left    = q[-1]
            # qm_left    = q[k] 
            # flux_left_U = flux_star(qp_left,qm_left,alpha,a)
            
            # Flux right
            qm_right   = q[k+N-1]
            qp_right   = q[k+N] 
            flux_right_U = flux_star(qp_right,qm_right,alpha,np.sqrt(a))

            # Flux for Q equation
            
            # Flux left 
            um_left     = u[k] 
            flux_left_Q  = np.sqrt(a)*um_left 
            # Flux right
            um_right   = u[k+N-1]
            up_right   = u[k+N] 
            flux_right_Q = flux_star(up_right,um_right,alpha,np.sqrt(a))

        elif k == (len(u)-N):

            # Flux for Uk equation
            
            # Flux left
            qm_left   = q[k]
            qp_left   = q[k-1] 
            flux_left_U = flux_star(qm_left,qp_left,alpha,np.sqrt(a))

            # Flux right (Reflective)
            qm_right      = q[-1] 
            qp_right      = -qm_right
            flux_right_U  = flux_star(qp_right,qm_right,alpha,np.sqrt(a))
            
            # Flux right (Periodic)
            # qm_right      = q[-1] 
            # qp_right      = q[0]
            # flux_right_U  = flux_star(qp_right,qm_right,alpha,np.sqrt(a))
            
            # Flux for Q equation
            # Flux left
            um_left   = u[k]
            up_left   = u[k-1] 
            flux_left_Q = flux_star(um_left,up_left,alpha,np.sqrt(a))

            # Flux right
            um_right      = u[-1] 
            flux_right_Q  = np.sqrt(a)*um_right

        else:

            # Flux for Uk equation
            # left boundary of element
            qm_left    = q[k]
            qp_left    = q[k-1] 
            flux_left_U  = flux_star(qm_left,qp_left,alpha,np.sqrt(a))
            
            # right boundary of element
            qm_right   = q[k+N-1]
            qp_right   = q[k+N]
            flux_right_U = flux_star(qp_right,qm_right,alpha,np.sqrt(a))

            # Flux for Q equation
            # left boundary of element
            um_left    = u[k]
            up_left    = u[k-1] 
            flux_left_Q  = flux_star(um_left,up_left,alpha,np.sqrt(a))
            
            # right boundary of element
            um_right   = u[k+N-1]
            up_right   = u[k+N]
            flux_right_Q = flux_star(up_right,um_right,alpha,np.sqrt(a))
           
        if formulation == "w":
            
            rhs_u = lagrange_rhs_right*(flux_right_U)-lagrange_rhs_left*(flux_left_U)
            rhs_q = lagrange_rhs_right*(flux_right_Q)-lagrange_rhs_left*(flux_left_Q)
            
            qk = Mk_inv@(-((S.T)@(np.sqrt(a)*uk))+rhs_q)
            DUDT[k:int(k+N)] = Mk_inv@(-((S.T)@(np.sqrt(a)*qk))+rhs_u)

        elif formulation == "s":
            
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
     
    x_left = -2
    x_right = 2
    Nm1 = 20
    N = Nm1+1
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
    a = 1.0
    alpha = 1.0 

    max_step = 0.0001
    t0 = 0.008
    tf = 0.03
    formulation = "s"
    u0 = u_exact(x_total,t0,a)

    sol = solve_ivp(f_func, [t0, tf], u0, args=(Mk_inv,Dx,S,N,alpha,a,formulation), max_step=max_step, dense_output=True, method="Radau")

    x_large = np.linspace(x_left,x_right,1000)

    plt.figure()
    plt.plot(x_total,sol.y[:,-1],'.',label=r"$u(x,t_f)$")
    plt.plot(x_large,u_exact(x_large,t0,a),'-',label=r"$u(x,t_0)$")
    plt.plot(x_large,u_exact(x_large,tf,a),label=r"$u_{exact}(x,t_f)$")
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

    #%%
    
    
    
    plt.figure()

    #plt.plot(x_large,u_exact(x_large,t0,a),'-',label=r"$u(x,t_0)$")
    plt.plot(x_large,u_exact(x_large,tf,a),label=r"$u_{exact}(x,t_f)$")
    
    uf = sol.y[:,-1]
    Vinv = np.linalg.inv(V)
    Next = 50
    
    for k in range(number_element):
        xk = x_total[k*N:(k+1)*N]
        xk_l = xk[0]
        xk_r = xk[-1]
        xk_ext = np.linspace(xk_l,xk_r,)
        r = 2 * (xk_ext - xk_l)/(xk_r - xk_l) - 1
        
        Vext,_,_ = legendre.vander(r,N)
        
        plt.plot(xk_ext,Vext@Vinv@uf[k*N:(k+1)*N],".")

    plt.legend() 


#%%
    I = np.eye(number_element)
    M_total = np.kron(I,Mk)
    
    diff = uf - u_exact(x_total,tf,a)
    
    error = np.sqrt(diff @ M_total @ diff)
    print(error) 

    #%% 

    # Define a fontsize variable
    fontsize = 12  # You can change this value to control the font size

    # Exact solution and error calculation
    U_exact = u_exact(X, T, a)
    U = sol.y.T
    
    # Error
    error = U - U_exact
    
    # Create a plot with 3 columns and 1 row
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Numerical solution plot
    im1 = axes[0].pcolormesh(T, X, U, shading='auto', cmap='viridis')
    axes[0].set_title("Numerical Solution", fontsize=fontsize)
    axes[0].set_xlabel(r"$t$: time", fontsize=fontsize)
    axes[0].set_ylabel(r"$x$: space", fontsize=fontsize)
    fig.colorbar(im1, ax=axes[0])
    
    # Exact solution plot
    im2 = axes[1].pcolormesh(T, X, U_exact, shading='auto', cmap='viridis')
    axes[1].set_title("Exact Solution", fontsize=fontsize)
    axes[1].set_xlabel(r"$t$: time", fontsize=fontsize)
    axes[1].set_ylabel(r"$x$: space", fontsize=fontsize)
    fig.colorbar(im2, ax=axes[1])
    
    # Error plot
    im3 = axes[2].pcolormesh(T, X, error, shading='auto', cmap='viridis')
    axes[2].set_title("Error", fontsize=fontsize)
    axes[2].set_xlabel(r"$t$: time", fontsize=fontsize)
    axes[2].set_ylabel(r"$x$: space", fontsize=fontsize)
    fig.colorbar(im3, ax=axes[2])
    
    # Add a main title with controlled fontsize
    fig.suptitle("Diffusion Problem", fontsize=fontsize+2)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Adjust space for the suptitle
    
    # Display the plot
    plt.show()





