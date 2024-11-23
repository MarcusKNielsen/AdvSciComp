import numpy as np
import func.legendre as legendre

def flux_star(um,up,alpha,a):
        flux = a*(up+um)/2 + np.abs(a)*(1-alpha)/2*(up-um)
        return flux

def f_func(t,u,Mk_inv,S,N,alpha,a,g0_val,formulation):
    
    DUDT = np.zeros_like(u)
    
    lagrange_rhs_left = np.zeros(N)
    lagrange_rhs_left[0] = 1 
    lagrange_rhs_right = np.zeros(N)
    lagrange_rhs_right[-1] = 1

    for k in range(0,len(u),N):

        uk = u[k:k+N]
        
        if k == 0:
            
            # left boundary of element
            up_left    = g0_val(t)
            um_left   = u[k] 
            flux_left  = a*up_left 
            
            # right boundary of element
            um_right   = u[k+N-1]
            up_right   = u[k+N] 
            flux_right = flux_star(up_right,um_right,alpha,a)
            
            if formulation == "w":
                rhs = lagrange_rhs_right*(flux_right)-lagrange_rhs_left*(flux_left)
            elif formulation == "s":
                rhs = lagrange_rhs_right*(a*um_right-flux_right)-lagrange_rhs_left*(a*um_left-flux_left)

        elif k == (len(u)-N):
            
            # left boundary of element
            um_left    = u[k]
            up_left    = u[k-1] 
            flux_left  = flux_star(um_left,up_left,alpha,a)
            
            # right boundary of element
            um_right   = u[-1]
            flux_right = a*um_right
            
            if formulation == "w":
                rhs = lagrange_rhs_right*(flux_right)-lagrange_rhs_left*(flux_left)
            elif formulation == "s":
                rhs = lagrange_rhs_right*(a*um_right-flux_right)-lagrange_rhs_left*(a*um_left-flux_left)

        else:

            # left boundary of element
            um_left    = u[k]
            up_left    = u[k-1] 
            flux_left  = flux_star(um_left,up_left,alpha,a)
            
            # right boundary of element
            um_right   = u[k+N-1]
            up_right   = u[k+N]
            flux_right = flux_star(up_right,um_right,alpha,a)

            if formulation == "w":
                rhs = lagrange_rhs_right*(flux_right)-lagrange_rhs_left*(flux_left)
            elif formulation == "s":
                rhs = lagrange_rhs_right*(a*um_right-flux_right)-lagrange_rhs_left*(a*um_left-flux_left)

        if formulation == "w":
            DUDT[k:int(k+N)] = Mk_inv@(((S.T)@(a*uk))-rhs) 
        elif formulation == "s":
            DUDT[k:int(k+N)] = Mk_inv@(-S@(a*uk)+rhs) 

    return DUDT

def total_grid_points(number_element,x_nodes,a,b):

    k_boundary = np.linspace(a,b,(number_element+1))
    x_total = np.zeros(len(x_nodes)*number_element)
    x_total = np.array([(x_nodes+1)/2*(xr-xl)+xl for xl,xr in zip(k_boundary[:-1],k_boundary[1:])])

    return x_total.ravel()

def g_func(x,t,a):
    return np.sin(np.pi*(x-a*t))

def g0_val(t):
    return np.sin(np.pi*(-1-a*t))

if __name__ == "__main__":
    
    from scipy.integrate import solve_ivp
    import matplotlib.pyplot as plt
    
    x_left = -1
    x_right = 1
    N = 5
    number_element = 10
    x_nodes = legendre.nodes(N)
    x_total = total_grid_points(number_element,x_nodes,x_left,x_right)
    u0 =  np.sin(np.pi*x_total) # scipy.stats.norm.pdf(x_total,scale=0.1)  #
    h = (x_total[-1]-x_total[0])/number_element
    
    V,Vx,w = legendre.vander(x_nodes)
    M = np.linalg.inv(V@V.T)
    Mk = (h/2)*M
    Mk_inv = np.linalg.inv(Mk) 
    Dx = Vx@np.linalg.inv(V)
    S = M@Dx
    a = 1.0
    alpha = 1.0 
    max_step = 0.01 
    tf = 10.2 
    formulation = "w" 
    
    sol = solve_ivp(f_func, [0, tf], u0, args=(Mk_inv,S,N,alpha,a,g0_val,formulation), max_step=max_step, dense_output=True, method="RK23")
    
    plt.figure()
    plt.plot(x_total,sol.y[:,-1],".",label=r"$u(x,t_f)$")
    plt.plot(x_total,sol.y[:,0],label=r"$u(x,t_0)$")
    #plt.plot(x_total,g_func(x_total,sol.t[-1],a),label=r"$u_{exact}$")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    
    plt.figure()
    X, T = np.meshgrid(x_total, sol.t)
    
    # Create the pcolormesh plot
    pcm = plt.pcolormesh(T, X, sol.y.T)
    
    # Label the axes and add a title
    plt.xlabel("t: time")
    plt.ylabel("x: space")
    plt.title("Advection Equation")
    
    # Add the colorbar
    plt.colorbar(pcm, label="u(x,t)")
    
    plt.show()
    
    print(f"error = {np.max(np.abs(g_func(x_total,sol.t[-1],a) - sol.y[:,-1]))}")


