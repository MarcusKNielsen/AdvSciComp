import numpy as np
from fourier import nodes, diff_matrix
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from L2space import discrete_inner_product
from scipy.fft import fft, ifft, fftfreq

def u_exact(x,t,c,x0):
    return 0.5*c*1/(np.cosh(0.5*np.sqrt(c)*(x-c*t-x0)))**2

def Dealias(uhat, vhat, N, M):
    
    # Python Implementation of slide 23 Lecture_6_Nonlinear
    
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
    
    return what

# Parameters in problem
x1 = 20
x2 = 20
c  = 0.25
x0 = 0
t0 = 0
a = 1.0 # 2*np.pi/(x1+x2)

N = 100
M = 2*N
k = fftfreq(N,2*np.pi/N)

w = nodes(N)
x  = w*(x1+x2)/(2*np.pi) - x1

u0 = u_exact(x,t0,c,x0)
u0hat1 = fft(u0)

# Initial condition
K = 300
w_large = nodes(K)
x_large = w_large*(x1+x2)/(2*np.pi) - x1
u0 = u_exact(x_large,t0,c,x0)
u0hat = fft(u0)
u0hat = np.concatenate([u0hat[:N//2], u0hat[(K - N//2):]]) * (N/K)

#%%

def f(t,uhat,N,M,k,a):
    vhat = 1j*k*uhat
    what = Dealias(uhat, vhat, N, M)
    return -6*a*what + 1j*(k*a)**3*uhat

tf = 5.0
max_step = 0.1
sol = solve_ivp(f,[0, tf],u0hat,args=(N,M,k,a),max_step=max_step,dense_output=True,method="RK23")


t = sol.t
U = sol.y.T

uf = ifft(U[-1]).real
uf_exact = u_exact(x,tf,c,x0)

plt.figure()
plt.plot(x_large,u0,label="u0")
plt.plot(x,uf_exact,label="exact")
plt.plot(x,uf,label="approx")
plt.legend()
plt.show()

plt.figure()
plt.plot(x,np.abs(uf_exact - uf))
plt.show()



