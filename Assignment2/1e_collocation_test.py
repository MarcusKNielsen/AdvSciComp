import numpy as np
import sys
sys.path.insert(0,r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\2_semester\Advanced nummerical\AdvSciComp\Assignment2\func")
from fourier import nodes, diff_matrix
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from L2space import discrete_inner_product
from scipy.fft import fft, ifft, fftfreq

def u_exact(x,t,c,x0):
    return 0.5*c*1/(np.cosh(0.5*np.sqrt(c)*(x-c*t-x0)))**2

def Dealias(u, v, N, M):
    
    # Python Implementation of slide 23 Lecture_6_Nonlinear
    
    # FFT
    uhat = fft(u)
    vhat = fft(v)
    
    # Padding the uhat and vhat arrays
    uhatpad = np.concatenate([uhat[:N//2], np.zeros(M - N), uhat[N//2:]])
    vhatpad = np.concatenate([vhat[:N//2], np.zeros(M - N), vhat[N//2:]])
    
    # Inverse FFT to get upad and vpad
    upad = ifft(uhatpad)
    vpad = ifft(vhatpad)
    
    # Pointwise multiplication in physical space
    wpad = upad * vpad
    
    # Forward FFT to get wpad_hat
    wpad_hat = fft(wpad)
    
    # Dealiasing step
    what = (M/N) * np.concatenate([wpad_hat[:N//2], wpad_hat[M - N//2:]])
    
    # Inverse FFT
    w = ifft(what)
    
    return w.real

# Parameters in problem
x1 = 20
x2 = 20
c  = 1.0
x0 = 0
t0 = 0
a = 2*np.pi/(x1+x2)

N = 200
M = 3*N//2
D = diff_matrix(N)
D3 = D @ D @ D

w = nodes(N)
x  = w*(x1+x2)/(2*np.pi) - x1

# Initial condition
u0 = u_exact(x,t0,c,x0)


#%%

def f(t,u,D,D3,a):
    return -6*a*u*D@u - a**3 * D3@u

def f_alias_free(t,u,D,D3,a,N,M):
    w = Dealias(u, D@u, N, M)
    return -6*a*w - a**3 * D3@u


tf = 5.0
max_step = 0.1

#sol = solve_ivp(f,[0, tf],u0,args=(D,D3,a),max_step=max_step,dense_output=True,method="RK23")
sol = solve_ivp(f_alias_free,[0, tf],u0,args=(D,D3,a,N,M),max_step=max_step,dense_output=True,method="RK23")

t = sol.t
U = sol.y.T

uf = U[-1]
uf_exact = u_exact(x,tf,c,x0)

plt.figure()
plt.plot(x,u0,label="u0")
plt.plot(x,uf_exact,label="exact")
plt.plot(x,uf,label="approx")
#plt.title("Collocation ignoring alias errors")
plt.legend()
plt.show()

plt.figure()
plt.plot(x,np.abs(uf_exact - uf))
plt.show()



