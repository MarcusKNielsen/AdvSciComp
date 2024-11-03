#%% Importing modules
import numpy as np
import sys
sys.path.insert(0,r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\2_semester\Advanced nummerical\AdvSciComp\Assignment2\func")
from fourier import nodes, diff_matrix
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import functions_TIME as functions

dealias = True

#%% f)

# Constants
N = 50
# Fine grid (zero-padding)
M = 3*N//2
c_value1 = 0.25
c_value2 = 0.5
tf = 120.0
alpha = 0.9
x0_1 = -15
x0_2 = -40
x1 = 45
x2 = 30
w = nodes(N)
D = diff_matrix(N)
D3 = D @ D @ D
a = 2 * np.pi / (x1 + x2)

# Initial condition
x  = w*(x1+x2)/(2*np.pi) - x1

# MANGLER DEALIAS IC HER
u0 = functions.u_exact(x, 0, c_value1, x0_1)+functions.u_exact(x, 0, c_value2, x0_2)

# Time integration
max_step = 0.1 #alpha * 1.73 * 8 / (N**3 * a**3) 

if dealias:
    sol = solve_ivp(functions.f_alias_free,[0, tf],u0,args=(D,D3,a,N,M),max_step=max_step,dense_output=True,method="RK23")
else:
    sol = solve_ivp(functions.f, [0, tf], u0, args=(D, D3), max_step=max_step,dense_output=True, method="RK23")

t = sol.t
U = sol.y.T

plt.figure()
X,T = np.meshgrid(x,t)

plt.pcolormesh(T,X,U)
plt.xlabel("t: time")
plt.ylabel("x: space")
plt.title(f"Diffusion Equation")

plt.show()
