
#%% 
import numpy as np
import sys
sys.path.insert(0,r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\2_semester\Advanced nummerical\AdvSciComp\Assignment2\func")
from fourier import nodes
import matplotlib.pyplot as plt
from L2space import discrete_inner_product
from scipy.fft import fft, ifft

def compute_L2_error(numerical, exact, weights):
    return np.sqrt(discrete_inner_product(numerical - exact, numerical - exact, weights))

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

def f(t,u,D,D3,x1=20,x2=20):
    a = 2*np.pi/(x1+x2)
    return -6*a*u*D@u - a**3 * D3@u

def f_alias_free(t,u,D,D3,a,N,M):
    w = Dealias(u, D@u, N, M)
    return -6*a*w - a**3 * D3@u

def u_exact(x,t,c,x0,x1=20,x2=20):
    a = 2*np.pi/(x1+x2)
    return 0.5*c*1/(np.cosh(0.5*np.sqrt(c)*(x-c*t-x0)))**2

def dealias_IC(N,M,w0,x1,x2,c_value):
    w_large = nodes(M)
    x_large = w_large*(x1+x2)/(2*np.pi) - x1 
    x0 = w0*(x1+x2)/(2*np.pi) - x1
    u0 = u_exact(x_large, 0, c_value, x0)
    u0_hat = fft(u0)
    u0_hat = (N/M)*np.concatenate([u0_hat[:N//2], u0_hat[M - N//2:]])
    u0 = ifft(u0_hat)

    return u0.real

dealias = True

#%% 

if __name__ == "__main__":
    N = 40
    M = 3*N//2
    c_value = 1.0
    x1 = 20
    x2 = 20
    w_small = nodes(N)
    w_large = nodes(M)
    x_small = w_small*(x1+x2)/(2*np.pi) - x1 
    x_large = w_large*(x1+x2)/(2*np.pi) - x1 
    w0 = 2
    x0 = w0*(x1+x2)/(2*np.pi) - x1
    u0 = u_exact(x_large, 0, c_value, x0)
    u0_hat = fft(u0)
    u0_hat = (N/M)*np.concatenate([u0_hat[:N//2], u0_hat[M - N//2:]])
    u0_new = ifft(u0_hat)

    plt.figure()
    plt.plot(x_small,u0_new,label="u0 new")
    plt.plot(x_large,u0,label="u0")
    plt.legend()
    plt.show()



