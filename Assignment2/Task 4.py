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

def JacobiP(x,alpha,beta,N):

    if x.size > 1:
        J_nm2 = np.ones(len(x)) # J_{n-2}
    else:
        J_nm2 = 1
    J_nm1 = 1/2*(alpha-beta+(alpha+beta+2)*x) # J_{n-1}

    if N==0:
        return J_nm2
    elif N==1:
        return J_nm1

    for n in range(1,N):

        # Computing the recursive coefficients
        anm2  = 2*(n+alpha)*(n+beta)/( (2*n+alpha+beta+1)*(2*n+alpha+beta) )
        anm1  = (alpha**2-beta**2)/( (2*n+alpha+beta+2)*(2*n+alpha+beta) )
        an    = 2*(n+1)*(n+beta+alpha+1)/( (2*n+alpha+beta+2)*(2*n+alpha+beta+1) )

        # Computing
        J_n = ( (anm1 + x )*J_nm1 - anm2*J_nm2 ) / an

        # Updating step
        J_nm2 = J_nm1
        J_nm1 = J_n
    
    return J_n

def get_extended_vandermonde(x,N):
    K = len(x)
    Vm = np.zeros([K,N])
    for j in range(N):
        Vm[:,j] = JacobiP(x,alpha=0,beta=0,N=j)
    return Vm

def BC(A,b,D,BC_method):

    if BC_method == strong:
        A[0] = 0
        A[-1] = 0
        A[0,0] = 1 
        A[-1,-1] = 1 

        b[0] = u_exact(-1)
        b[-1] = u_exact(1)

    elif BC_method == week:
        A[0,0] += tau 
        A[-1,-1] += tau 
        A[0] += D[0]
        A[-1] -= D[-1]

        b[0] += tau*u_exact(-1)
        b[-1] += tau*u_exact(1) 
    
    return A,b

def construct_system(N):
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

def Shifted_BC(d,BC_method,N=20):

    A,b,D,x_GL,V = construct_system(N)
    A,b = BC(A,b,D,BC_method=BC_method)
    u_weak = np.linalg.solve(A,b)

    x_lin = np.linspace(-1-d,1+d,100)
    V_ext,_,_ = legendre.vander(x_lin,N)

    u_sol = V_ext@np.linalg.inv(V)@u_weak

    return u_sol,x_lin 

N=30
N_list = np.arange(10,N,2)

strong,week = True,False
tau = 1

error_weak = []
error_strong = []

for n in N_list:

    # Weak 
    A,b,D,x_GL,V = construct_system(n)
    A,b = BC(A,b,D,BC_method=week)
    u_weak = np.linalg.solve(A,b)
    # Strong
    A,b,D,x_GL,V = construct_system(n)
    A,b = BC(A,b,D,BC_method=strong)
    u_strong = np.linalg.solve(A,b)

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
A,b = BC(A,b,D,BC_method=week)
u_weak = np.linalg.solve(A,b)

# Strong
A,b,D,x_GL,V = construct_system(N=6)
A,b = BC(A,b,D,BC_method=strong)
u_strong = np.linalg.solve(A,b)

# Linspace
x_lin = np.linspace(-1,1,100)
plt.figure()
plt.plot(x_GL,u_weak,label="Weak-CBM")
plt.plot(x_GL,u_strong,label="Strong-CBM")
plt.plot(x_lin,u_exact(x_lin),label="$u_{exact}$")
plt.legend()
plt.xlabel("x")


#%% Shifted boundary 

u_shifted,x_lin = Shifted_BC(0.5,week)

# test figure
plt.figure()
plt.plot(x_lin,u_shifted,label="Weak-SBM")
plt.plot(x_lin,u_exact(x_lin),label="$u_{exact}$")
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
        u_shifted,x_lin = Shifted_BC(d,week,N=n)

        error_shifted.append(np.max(np.abs(u_shifted-u_exact(x_lin))))
    
    # Plotting for each d
    plt.loglog(N_list,error_shifted,"-o",label=f"Weak-SBM, d={d}")

plt.legend()
plt.xlabel("N")

#%% All 7 convergence studies

d_list = np.array([0.5,0.25,0,-0.25,-0.5])

plt.figure()
for d in d_list:

    error_shifted = []

    for n in N_list:

        # Weak 
        u_shifted,x_lin = Shifted_BC(d,week,N=n)

        error_shifted.append(np.max(np.abs(u_shifted-u_exact(x_lin))))
    
    # Plotting for each d
    plt.loglog(N_list,error_shifted,"-o",label=f"Weak-SBM, d={d}")

plt.loglog(N_list,error_weak,"-o",label="Weak-CBM")
plt.loglog(N_list,error_strong,"-o",label="Strong-CBM")
plt.legend()
plt.xlabel("N")
plt.show()







