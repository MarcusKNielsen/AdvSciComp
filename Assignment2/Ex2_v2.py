
#%%
import numpy as np
import sys
sys.path.insert(0,r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\2_semester\Advanced nummerical\AdvSciComp\Assignment2\func")
from fourier import nodes, diff_matrix
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import norm
from L2space import discrete_inner_product
from scipy.fft import fft, ifft, fftfreq
import scipy

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


#%% Opgave c)

def f(t,u,D,D3,x1=20,x2=20):
    a = 2*np.pi/(x1+x2)
    return -6*a*u*D@u - a**3 * D3@u

def f_alias_free(t,u,D,D3,a,N,M):
    w = Dealias(u, D@u, N, M)
    return -6*a*w - a**3 * D3@u

def u_exact2(w,t,c,w0,x1=20,x2=20):
    a = 2*np.pi/(x1+x2)
    x  = w*(x1+x2)/(2*np.pi) - x1
    x0 = w0*(x1+x2)/(2*np.pi) - x1
    return 0.5*c*1/(np.cosh(0.5*np.sqrt(c)*(x-c*t-x0)))**2

def u_exact(x,t,c,x0,x1=20,x2=20):
    a = 2*np.pi/(x1+x2)
    return 0.5*c*1/(np.cosh(0.5*np.sqrt(c)*(x-c*t-x0)))**2

def u_exact3(x,t,c,x0,x1=20,x2=20):
    a = 2*np.pi/(x1+x2)
    w  = 2*np.pi*(x+x1)/(x1+x2)
    w0 = 2*np.pi*(x0+x1)/(x1+x2)
    return 0.5*c*1/(np.cosh(0.5*np.sqrt(c)*(w-c*t-w0)))**2

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



#%% Convergence test:
# Set up different values for N (number of grid points)
N_values = np.arange(4,50,6)
errors = []

# Constants
c_value = 1.0
tf = 1e-3
alpha = 0.01
x1, x2 = 20, 20
w0 = np.pi

for N in N_values:

    # Fine grid (zero-padding)
    M = 3*N//2

    # Grid points and differentiation matrices
    w = nodes(N)
    D = diff_matrix(N)
    D3 = D @ D @ D
    a = 2 * np.pi / (x1 + x2)

    # Initial condition dealiasing
    if dealias:
        x = w*(x1+x2)/(2*np.pi) - x1 
        u0 = dealias_IC(N,M,w0,x1,x2,c_value)
    else:
        x = w*(x1+x2)/(2*np.pi) - x1 
        x0 = w0*(x1+x2)/(2*np.pi) - x1
        u0 = u_exact(x, 0, c_value, x0)
    
    # Time integration
    max_step = 1e-6 #alpha * 1.73 * 8 / (N**3 * a**3)

    if dealias:
        sol = solve_ivp(f_alias_free,[0, tf],u0,args=(D,D3,a,N,M),max_step=max_step,dense_output=True,method="RK23")
    else:
        sol = solve_ivp(f, [0, tf], u0, args=(D, D3), max_step=max_step,dense_output=True, method="RK23")

    # Extract solution at final time 
    U_approx = sol.y[:, -1]

    # Exact solution at final time
    x  = w*(x1+x2)/(2*np.pi) - x1
    x0 = w0*(x1+x2)/(2*np.pi) - x1
    U_exact = u_exact(x, tf, c_value, x0)

    # Compute the L2 error
    error = np.max(np.abs(U_approx-U_exact))
    errors.append(error)

plt.figure()
plt.loglog(N_values,errors)
plt.xticks(np.arange(4,50,6))
plt.yticks([1e-1,1e-3,1e-5,1e-7,1e-10])
plt.minorticks_on()

#%% Opgave d)
N = 40
# Fine grid (zero-padding)
M = 3*N//2
w = nodes(N)
D = diff_matrix(N)
D3 = D @ D @ D
x1, x2 = 20, 20
a = 2 * np.pi / (x1 + x2)

c = np.array([0.25,0.5,1])

alpha = 0.01
max_step = alpha * 1.73*8/(N**3*a**3)

for c_i in c:

    if dealias:
        x = w*(x1+x2)/(2*np.pi) - x1 
        u0 = dealias_IC(N,M,w0,x1,x2,c_value)
    else:
        x = w*(x1+x2)/(2*np.pi) - x1 
        x0 = w0*(x1+x2)/(2*np.pi) - x1
        u0 = u_exact(x, 0, c_value, x0)

    tf = 1.0
    if dealias:
        sol = solve_ivp(f_alias_free,[0, tf],u0,args=(D,D3,a,N,M),max_step=max_step,dense_output=True,method="RK23")
    else:
        sol = solve_ivp(f, [0, tf], u0, args=(D, D3), max_step=max_step,dense_output=True, method="RK23")

    t = sol.t
    U = sol.y.T
    plt.figure()
    X,T = np.meshgrid(x,t)

    plt.pcolormesh(T,X,U)
    plt.xlabel("t: time")
    plt.ylabel("x: space")
    plt.title(f"Diffusion Equation: c={c_i}")

    plt.figure()
    plt.plot(x,U[0],".-",label=r"$u(x,0) (Approx)$")
    plt.plot(x,U[-1],".-",label=r"$u(x,T)$ (Approx)")
    plt.plot(x,u_exact(x,tf,c_i,x0),"--",label=r"$u(x,T)$ (Exact)")
    plt.plot(x,u_exact(x,0,c_i,x0),"--",label=r"$u(x,0)$ (Exact)")
    plt.title(f"c={c_i}")
    plt.legend()

    int_M_test = np.zeros(len(t))
    int_V_test = np.zeros(len(t))
    int_E_test = np.zeros(len(t))
    L2_error = np.zeros(len(t))   # L2-norm error over time
    Linf_error = np.zeros(len(t)) # Linf-norm error over time
    
    for i in range(len(t)):
        weight = np.ones_like(x)*2*np.pi/N
        int_M_test[i] = discrete_inner_product(np.ones_like(U[i]),U[i],weight)
        int_V_test[i] = discrete_inner_product(U[i],U[i],weight)
        int_E_term1   = discrete_inner_product(D@U[i],D@U[i],weight)
        int_E_term2   = discrete_inner_product(U[i],U[i]*U[i],weight)
        int_E_test[i] = 0.5*int_E_term1 - int_E_term2

        # Numerical solution at time t[i]
        U_approx = U[i]
        
        # Exact solution at time t[i]
        x  = w*(x1+x2)/(2*np.pi) - x1
        x0 = w0*(x1+x2)/(2*np.pi) - x1
        U_exact = u_exact(x, t[i], c_i, x0)
        
        # Compute L2 norm of the error
        L2_error[i] = np.sqrt(discrete_inner_product(U_approx - U_exact, U_approx - U_exact, weight))
        
        # Compute Linf norm of the error (max absolute error)
        Linf_error[i] = np.max(np.abs(U_approx - U_exact))
    
    
    plt.figure()
    plt.plot(t,int_M_test,label=r"M= $\int u(x,t) dx$")
    plt.plot(t,int_V_test,label=r"V= $\int u(x,t)^2 dx$")
    plt.plot(t,int_E_test,label=r"E= $\int \frac{1}{2}u_x(x,t)^2 - u^3 dx$")
    plt.title(f"Mass (M), Momentum (V), Energy (E), for c={c_i}")
    plt.xlabel("t")
    plt.legend()
    
    # Plot the L2 norm error evolution over time
    plt.figure()
    plt.plot(t, L2_error, label=r"$\|u - \mathcal{I}_N u\|_{L_2}$")
    plt.xlabel('Time (t)')
    plt.ylabel(r"$L_2$-norm Error")
    plt.title(f'Evolution of $L_2$ Error for c={c_i}')
    plt.legend()
    plt.grid(True)
    

    # Plot the Linf norm error evolution over time
    plt.figure()
    plt.plot(t, Linf_error, label=r"$\|u - \mathcal{I}_N u\|_{L_\infty}$", color="red")
    plt.xlabel('Time (t)')
    plt.ylabel(r"$L_\infty$-norm Error")
    plt.title(f'Evolution of $L_\infty$ Error for c={c_i}')
    plt.legend()
    plt.grid(True)
    


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
x_lin = nodes(500)
N1 = len(x)
N2 = len(x_lin)

for c_i in c:

    if dealias:
        x = w*(x1+x2)/(2*np.pi) - x1
        u0 = dealias_IC(N,M,w0,x1,x2,c_value)
    else:
        x = w*(x1+x2)/(2*np.pi) - x1 
        x0 = w0*(x1+x2)/(2*np.pi) - x1
        u0 = u_exact(x, 0, c_value, x0)

    tf = 1.0
    
    if dealias:
        sol = solve_ivp(f_alias_free,[0, tf],u0,args=(D,D3,a,N,M),max_step=max_step,dense_output=True,method="RK23")
    else:
        sol = solve_ivp(f, [0, tf], u0, args=(D, D3), max_step=max_step,dense_output=True, method="RK23")

    U_approx = sol.y[:, -1]

    uk_approx = fft(U_approx)
    uk_exact = fft(u_exact(x_lin,tf,c_i,x0))[int(len(x_lin)/2-N):int(len(x_lin)/2+N)]

    uk1 = scipy.fft.fft(uk_approx)
    dx1 = (x[1:] - x[:-1])[0]
    freq1 = scipy.fft.fftfreq(N1, d=dx1)

    uk2 = scipy.fft.fft(uk_exact)
    dx2 = (x_lin[1:] - x_lin[:-1])[0]
    freq2 = scipy.fft.fftfreq(N2, d=dx2)

    uk = np.zeros_like(uk1)
    uk[:N1//2] = uk2[:N1//2]
    uk[N1//2:] = uk2[N1+N1//2:]
    uk = (N1/(2*N1))*uk

    plt.figure()
    plt.plot(freq1,np.abs(uk1.real),".",label=f"grid points: {N1}")
    plt.plot(freq1,np.abs(uk.real),".",label=f"grid points: {N2}")
    plt.ylabel(r"$|u_k|$")
    plt.xlabel(r"$k$")
    plt.title(f"c={c_i}")
    plt.legend()
    



#%% f)

# Constants
N = 50
# Fine grid (zero-padding)
M = 3*N//2
c_value1 = 0.25
c_value2 = 0.5
tf = 40.0
alpha = 0.01
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

u0 = u_exact(x, 0, c_value1, x0_1)+u_exact(x, 0, c_value2, x0_2)

# Time integration
max_step = 0.1 #alpha * 1.73 * 8 / (N**3 * a**3)

if dealias:
    sol = solve_ivp(f_alias_free,[0, tf],u0,args=(D,D3,a,N,M),max_step=max_step,dense_output=True,method="RK23")
else:
    sol = solve_ivp(f, [0, tf], u0, args=(D, D3), max_step=max_step,dense_output=True, method="RK23")


t = sol.t
U = sol.y.T

plt.figure()
X,T = np.meshgrid(x,t)

plt.pcolormesh(T,X,U)
plt.xlabel("t: time")
plt.ylabel("x: space")
plt.title(f"Diffusion Equation")

plt.show()


