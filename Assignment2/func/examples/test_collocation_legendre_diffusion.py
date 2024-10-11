import numpy as np
from numpy.linalg import inv
from legendre import vander, discrete_inner_product
import matplotlib.pyplot as plt
from JacobiGL import JacobiGL
from scipy.integrate import solve_ivp
from scipy.stats import norm
N = 128
x = JacobiGL(0,0,N)

V,Vx,w = vander(x)


Vinv = inv(V)

D = Vx @ Vinv

eps = 0.001


A = eps*D@D

# No-flux Boundary Condition
tol = 1e10
A[0]  =  tol*D[0]
A[-1] = -tol*D[-1]

def f(t,u,eps,A):
    return A@u


# Setup Initial condition
u0 = norm.pdf(x,scale=0.1)
dx = x[1:] - x[:-1]
u0 = u0/discrete_inner_product(np.ones_like(u0),u0,w)

sol = solve_ivp(f,[0, 100.0],u0,args=(eps,A),max_step=0.5,dense_output=True,method="Radau")

t = sol.t
U = sol.y.T

X,T = np.meshgrid(x,t)

plt.figure()
plt.pcolormesh(T,X,U)
plt.xlabel("t: time")
plt.ylabel("x: space")
plt.title("Diffusion Equation")
plt.show()


plt.figure()
plt.plot(x,U[0],".-",label="t=0")
plt.plot(x,U[-1],".-",label=f"t={t[-1]}")
plt.show()


int_test = np.zeros(len(t))
for i in range(len(t)):
    int_test[i] = discrete_inner_product(np.ones_like(U[i]),U[i],w)

plt.figure()
plt.plot(t,int_test,label=r"$\int u(x,t) dx$")
plt.xlabel("x")
plt.legend()
plt.show()


