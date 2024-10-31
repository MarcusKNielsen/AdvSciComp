import numpy as np 
import sys 

sys.path.insert(0,r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\2_semester\Advanced nummerical\AdvSciComp\Assignment2\func")
import legendre
import L2space 
import JacobiP
import matplotlib.pyplot as plt

def u_exact(x):
    return np.sin(np.pi*x)+0.5

f = lambda x: np.pi**2*np.sin(np.pi*x)

def get_extended_vandermonde(x,N):
    K = len(x)
    Vm = np.zeros([K,N])
    for j in range(N):
        Vm[:,j] = JacobiP(x,alpha=0,beta=0,N=j)
    return Vm

def BC(A,b,D,BC_method,condA=False,d=0):

    if BC_method == "strong":
        A[0] = 0
        A[-1] = 0
        A[0,0] = 1 
        A[-1,-1] = 1 

        b[0] = u_exact(-1)
        b[-1] = u_exact(1)

    elif BC_method == "week":
        A[0,0] += tau 
        A[-1,-1] += tau 
        A[0] += D[0]
        A[-1] -= D[-1]

        b[0] += tau*u_exact(-1)
        b[-1] += tau*u_exact(1)

    elif BC_method == "week_SH":

        # Interpolation matrix 
        N = A.shape[0]

        # None shifted grid
        x_GL = legendre.nodes(N)
        # None shifted vandermonde
        V,_,_ = legendre.vander(x_GL)

        # Shifted grid
        x_GLa,x_GLb = x_GL,x_GL
        x_GLa[0] = -1-d
        x_GLb[-1] = 1+d

        # Shifted vandermonde -1-d
        V_ext_a,_,_ = legendre.vander(x_GLa)
        Ia = V_ext_a@np.linalg.inv(V)

        # Shifted vandermonde 1+d
        V_ext_b,_,_ = legendre.vander(x_GLb)
        Ib = V_ext_b@np.linalg.inv(V)

        A[0] += tau*Ia[0]
        A[-1] += tau*Ib[-1] 
        A[0] += D[0]
        A[-1] -= D[-1]

        b[0] += tau*u_exact(-1-d)
        b[-1] += tau*u_exact(1+d)
    
    condA_val = np.linalg.cond(A)

    if condA:
        return A,b,condA_val
    else:
        return A,b

def construct_system(N,condA=False):
    # Magic nodes
    x_GL = legendre.nodes(N)

    # Vandermonde + diff vandermode
    V,Vx,w = legendre.vander(x_GL)
    # Differentiation matrix
    D = Vx@np.linalg.inv(V)
    # Mass function 
    M = np.linalg.inv(V@V.T)

    # System matrix A and right hand-side f:
    A = D.T@M@D
    b = M@f(x_GL)

    return A,b,D,x_GL,V

N=30
N_list = np.arange(10,N,2)

tau = 1

error_weak = []
error_strong = []

condA_week_list = []
condA_strong_list = []

for n in N_list:

    # Weak 
    A,b,D,x_GL,V = construct_system(n)
    A,b,condA_week = BC(A,b,D,BC_method="week",condA=True)
    u_weak = np.linalg.solve(A,b)
    # Strong
    A,b,D,x_GL,V = construct_system(n)
    A,b,condA_strong = BC(A,b,D,BC_method="strong",condA=True)
    u_strong = np.linalg.solve(A,b)

    condA_week_list.append(condA_week)
    condA_strong_list.append(condA_strong)
    error_weak.append(np.max(np.abs(u_weak-u_exact(x_GL))))
    error_strong.append(np.max(np.abs(u_strong-u_exact(x_GL))))

plt.figure()
plt.loglog(N_list,error_weak,"-o",label="Weak-CBM")
plt.loglog(N_list,error_strong,"-o",label="Strong-CBM")
plt.xlabel("N")
plt.legend()

#%% Solution with N=6

# Weak
A,b,D,x_GL,V = construct_system(N=6)
A,b = BC(A,b,D,BC_method="week")
u_weak = np.linalg.solve(A,b)

# Strong
A,b,D,x_GL,V = construct_system(N=6)
A,b = BC(A,b,D,BC_method="strong")
u_strong = np.linalg.solve(A,b)

# Linspace
x_lin = np.linspace(-1,1,100)

# Interpolation 
V_ext,_,_ = legendre.vander(x_lin,N=6)

# Plot
plt.figure()
plt.plot(x_lin,V_ext@np.linalg.inv(V)@u_weak,label="Weak-CBM")
plt.plot(x_lin,V_ext@np.linalg.inv(V)@u_strong,label="Strong-CBM")
plt.plot(x_lin,u_exact(x_lin),label="$u_{exact}$")
plt.legend()
plt.xlabel("x")


#%% Shifted boundary 

d = 0.25
A,b,D,x_GL,V = construct_system(N=20)
A,b = BC(A,b,D,BC_method="week_SH",d=d)
u_shifted = np.linalg.solve(A,b)
x_GL =  legendre.nodes(20)

x_lin = np.linspace(-1-d,1+d,100)
V_ext,_,_ = legendre.vander(x_lin,N=20)
I_ext = V_ext@np.linalg.inv(V)

# test figure
plt.figure()
plt.plot(x_lin,u_exact(x_lin),label="$u_{exact}$")
plt.plot(x_lin,I_ext@u_shifted,'--',label="Weak-SBM")
plt.legend()
plt.xlabel("x")
plt.title("Test of extended Vandermonde SBC")

#%% Convergence test of shifted boundary conditions

d_list = np.array([0.5,0.25,0,-0.25,-0.5])

plt.figure()
for d in d_list:

    error_shifted = []

    for n in N_list:

        # Weak 
        A,b,D,x_GL,V = construct_system(n)
        A,b = BC(A,b,D,BC_method="week_SH",d=d)
        u_shifted = np.linalg.solve(A,b)
        x_GL =  legendre.nodes(n)

        error_shifted.append(np.max(np.abs(u_shifted-u_exact(x_GL))))
    
    # Plotting for each d
    plt.loglog(N_list,error_shifted,"-o",label=f"Weak-SBM, d={d}")

plt.legend()
plt.xlabel("N")

#%% All 7 convergence studies

d_list = np.array([0.5,0.25,0,-0.25,-0.5])

fig, axs = plt.subplots(1, 2, figsize=(10, 8))
for d in d_list:

    error_shifted = []
    condA_shifted_list = []

    for n in N_list:

        # Weak 
        A,b,D,x_GL,V = construct_system(n)
        A,b,condA = BC(A,b,D,BC_method="week_SH",condA=True,d=d)
        u_shifted = np.linalg.solve(A,b)

        x_GL =  legendre.nodes(n)

        condA_shifted_list.append(condA)
        error_shifted.append(np.max(np.abs(u_shifted-u_exact(x_GL))))
    
    # Plotting for each d
    axs[0].loglog(N_list,error_shifted,"-o",label=f"Weak-SBM, d={d}")
    axs[1].loglog(N_list,condA_shifted_list,"-o",label=f"Weak-SBM, d={d}")

axs[0].loglog(N_list,error_weak,"-o",label="Weak-CBM")
axs[0].loglog(N_list,error_strong,"-o",label="Strong-CBM")
axs[0].legend()
axs[0].set_xlabel("N")

axs[1].loglog(N_list,condA_week_list,"-o",label="Weak-CBM")
axs[1].loglog(N_list,condA_strong_list,"-o",label="Stong-CBM")
axs[1].set_xlabel("N")
axs[1].set_ylabel("Cond(A)")

plt.show()







