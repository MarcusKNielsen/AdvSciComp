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

def f_func(t,u,Mk_inv,S,N,alpha,a):
    
    DUDT = np.zeros_like(u)

    for k in range(0,len(u),N):

        lagrange_rhs = np.zeros(N)
        lagrange_rhs_left = lagrange_rhs
        lagrange_rhs_left[0] = 1
        lagrange_rhs_right = lagrange_rhs
        lagrange_rhs_right[-1] = 1

        if k == 0:
            uk = u[k:int(k+N)]
            ukp1 = u[int(k+N):int(k+2*N)]
            rhs = lagrange_rhs_right*(a*uk[-1]-flux_star_right(uk,ukp1,alpha,a))-lagrange_rhs_left*(a*uk[0])

        elif k == (len(u)-N):
            ukm1 = u[int(k-N):k]
            uk = u[k:int(k+N)]
            rhs = lagrange_rhs_right*(a*uk[-1])-lagrange_rhs_left*(a*uk[0]-flux_star_left(uk,ukm1,alpha,a))
        
        else:
            ukm1 = u[int(k-N):k]
            uk = u[k:int(k+N)]
            ukp1 = u[int(k+N):int(k+2*N)]
            rhs = lagrange_rhs_right*(a*uk[-1]-flux_star_right(uk,ukp1,alpha,a))-lagrange_rhs_left*(a*uk[0]-flux_star_left(uk,ukm1,alpha,a))

        DUDT[k:int(k+N)] = Mk_inv@(-S@(a*uk)+rhs) 

    return DUDT

def total_grid_points(number_element,x_nodes,a,b):

    k_boundary = np.linspace(a,b,(number_element+1))
    x_total = np.zeros(len(x_nodes)*number_element)
    x_total = np.array([(x_nodes+1)/2*(xr-xl)+xl for xl,xr in zip(k_boundary[:-1],k_boundary[1:])])

    return x_total.ravel()

x_left = -1
x_right = 1
N = 15
number_element = 30
x_nodes = legendre.nodes(N)
x_total = total_grid_points(number_element,x_nodes,x_left,x_right)
u0 = np.sin(np.pi*x_total) #scipy.stats.norm.pdf(x_total,scale=0.1)  #  
h = (x_total[-1]-x_total[0])/number_element

V,Vx,_ = legendre.vander(x_nodes)
M = np.linalg.inv(V@V.T)
Mk = (h/2)*M
Mk_inv = np.linalg.inv(M) 
Dx = Vx@np.linalg.inv(V)
S = M@Dx
a = 1
alpha = 0
max_step = 0.01
tf = 1.0

sol = solve_ivp(f_func, [0, tf], u0, args=(Mk_inv,S,N,alpha,a), max_step=max_step, dense_output=True, method="RK23")

plt.figure()
plt.plot(x_total,sol.y[:,-1],label="last")
plt.plot(x_total,sol.y[:,0],label="first")
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

plt.show()
