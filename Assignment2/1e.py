#%% Importing modules
import numpy as np
import sys
sys.path.insert(0,r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\2_semester\Advanced nummerical\AdvSciComp\Assignment2\func")
from fourier import nodes, diff_matrix
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fft import fft
import scipy
import Assignment2.functions_TIME as functions

dealias = True

#%% e) Aliasing errors
N = 40
# Fine grid (zero-padding)
M = 3*N//2
w = nodes(N)
D = diff_matrix(N)
D3 = D @ D @ D
x1, x2 = 20, 20
a = 2 * np.pi / (x1 + x2)
alpha = 0.01
max_step = alpha * 1.73*8/(N**3*a**3)
x_lin = nodes(40)
N1 = len(x_lin)
N2 = len(x_lin)
w0 = np.pi

c = np.array([0.25,0.5,1])

for c_i in c:

    if dealias:
        x = w*(x1+x2)/(2*np.pi) - x1
        x0 = w0*(x1+x2)/(2*np.pi) - x1
        u0 = functions.dealias_IC(N,M,w0,x1,x2,c_i)
    else:
        x = w*(x1+x2)/(2*np.pi) - x1 
        x0 = w0*(x1+x2)/(2*np.pi) - x1
        u0 = functions.u_exact(x, 0, c_i, x0)

    tf = 1.0
    
    if dealias:
        sol = solve_ivp(functions.f_alias_free,[0, tf],u0,args=(D,D3,a,N,M),max_step=max_step,dense_output=True,method="RK23")
    else:
        sol = solve_ivp(functions.f, [0, tf], u0, args=(D, D3), max_step=max_step,dense_output=True, method="RK23")

    U_approx = sol.y[:, -1]

    uk_approx = fft(U_approx)
    uk_exact = fft(functions.u_exact(x_lin,tf,c_i,x0))[int(len(x_lin)/2-N):int(len(x_lin)/2+N)]

    uk1 = scipy.fft.fft(uk_approx)
    dx1 = (x[1:] - x[:-1])[0]
    freq1 = scipy.fft.fftfreq(N1, d=dx1)

    uk2 = scipy.fft.fft(uk_exact)
    dx2 = (x_lin[1:] - x_lin[:-1])[0]
    freq2 = scipy.fft.fftfreq(N2, d=dx2)

    uk = np.zeros_like(uk2)
    uk[:N1//2] = uk2[:N1//2]
    uk[N1//2:] = uk2[N1+N1//2:]
    uk = (N1/(2*N1))*uk

    plt.figure()
    plt.plot(freq1,np.abs(uk1.real),".",label=f"grid points: {N1}")
    #plt.plot(freq1,np.abs(uk.real),".",label=f"grid points: {N2}")
    plt.ylabel(r"$|u_k|$")
    plt.xlabel(r"$k$")
    plt.title(f"c={c_i}")
    plt.legend()
    
plt.show()

