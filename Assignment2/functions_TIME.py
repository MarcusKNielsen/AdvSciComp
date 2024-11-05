import numpy as np
import sys
sys.path.insert(0,r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\2_semester\Advanced nummerical\AdvSciComp\Assignment2\func")
from fourier import nodes
from L2space import discrete_inner_product,discrete_L2_norm
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

def f(t,u,D,D3,a):
    return -6*a*u*(D@u) - a**3 * (D3@u)

def f_alias_free(t,u,D,D3,a,N,M):
    w = Dealias(u, D@u, N, M)
    return -6*a*w - a**3 * (D3@u)

def u_exact(x,t,c,x0,x1=20,x2=20):
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

#%%

if __name__ == "__main__":
    
    from fourier import nodes, diff_matrix
    from scipy.integrate import solve_ivp
    import matplotlib.pyplot as plt
    import matplotlib
    
    # Remove PGF backend configuration and LaTeX rendering options
    # (No need for matplotlib.use("pgf") or pgf-specific rcParams)
    
    dealias = False
    
    # Parameters
    N = 50
    c_values = [0.25, 0.5, 1.0]
    tf = 1.0
    alpha = 0.5
    x1, x2 = 40, 40
    w0 = np.pi
    M = 3*N//2
    
    # Font size variables
    title_fontsize = 10
    label_fontsize = 9
    tick_fontsize = 10
    
    # Set up a 2-row, 2-column subplot figure
    fig, axs = plt.subplots(2, 2, dpi=150)
    fig.set_size_inches(w=6.0, h=6.0)
    
    for idx, c_value in enumerate(c_values):
        # Grid points and differentiation matrices
        w = nodes(N)
        D = diff_matrix(N)
        D3 = D @ D @ D
        a = 2 * np.pi / (x1 + x2)
        
        # Initial condition dealiasing
        if dealias:
            x = w * (x1 + x2) / (2 * np.pi) - x1
            u0 = dealias_IC(N, M, w0, x1, x2, c_value)
        else:
            x = w * (x1 + x2) / (2 * np.pi) - x1
            x0 = w0 * (x1 + x2) / (2 * np.pi) - x1
            u0 = u_exact(x, 0, c_value, x0)
    
        # Time integration
        max_step = alpha * 1.73 * 8 / (N**3 * a**3)
    
        if dealias:
            sol = solve_ivp(f_alias_free, [0, tf], u0, args=(D, D3, a, N, M), max_step=max_step, dense_output=True, method="RK23")
        else:
            sol = solve_ivp(f, [0, tf], u0, args=(D, D3, a), max_step=max_step, dense_output=True, method="RK23")
    
        # Extract solution at final time
        U_approx = sol.y[:, -1]
    
        # Exact solution at final time
        N_large = 1000
        w_large = nodes(N_large)
        x_large = w_large * (x1 + x2) / (2 * np.pi) - x1
        x0 = w0 * (x1 + x2) / (2 * np.pi) - x1
        U_exact = u_exact(x_large, tf, c_value, x0)
        
        # Visualize on a larger grid
        # Fourier transform to obtain Fourier coefficients (spectral space)
        u_hat = fft(U_approx)

        # Zero-pad the Fourier coefficients to increase resolution
        u_hat_padded = np.zeros(N_large, dtype=complex)
        u_hat_padded[:N//2] = u_hat[:N//2]
        u_hat_padded[-N//2:] = u_hat[-N//2:]

        # Inverse Fourier transform to get the interpolated function on a finer grid
        u_interp = ifft(u_hat_padded) * (N_large / N)
        
        # Plot on the respective subplot
        row, col = divmod(idx, 2)
        axs[row, col].plot(x_large, U_exact, label=r"$u(x,T)$")
        axs[row, col].plot(x_large, u_interp.real,"--", label=r"$u_N(x,T)$")
        axs[row, col].set_title(f"Solution for c={c_value} with N={N}", fontsize=title_fontsize)
        axs[row, col].set_xlabel("x", fontsize=label_fontsize)
        axs[row, col].set_ylabel(r"$u(x,T)$", fontsize=label_fontsize)
        axs[row, col].set_xlim([-15, 15])
        axs[row, col].tick_params(axis='both', which='major', labelsize=tick_fontsize)
        axs[row, col].legend(fontsize=label_fontsize, loc="upper left")
    
    # Convergence plot in the fourth subplot (2nd row, 2nd column)
    ax_convergence = axs[1, 1]
    N_max = 150
    N_values = np.arange(10, N_max, 4)  # Range of N values for convergence testing
    
    for c_value in c_values:
        errors = []  # List to store errors for the current c_value
    
        for N in N_values:
            # Fine grid (zero-padding) 
            M = 3 * N // 2
    
            # Grid points and differentiation matrices
            w = nodes(N)
            D = diff_matrix(N)
            D3 = D @ D @ D
            a = 2 * np.pi / (x1 + x2)
    
            # Initial condition dealiasing
            if dealias:
                x = w * (x1 + x2) / (2 * np.pi) - x1
                u0 = dealias_IC(N, M, w0, x1, x2, c_value)
            else:
                x = w * (x1 + x2) / (2 * np.pi) - x1
                x0 = w0 * (x1 + x2) / (2 * np.pi) - x1
                u0 = u_exact(x, 0, c_value, x0)
    
            # Time integration
            max_step = alpha * 1.73 * 8 / (N**3 * a**3)
    
            if dealias:
                sol = solve_ivp(f_alias_free, [0, tf], u0, args=(D, D3, a, N, M), max_step=max_step, dense_output=True, method="RK23")
            else:
                sol = solve_ivp(f, [0, tf], u0, args=(D, D3, a), max_step=max_step, dense_output=True, method="RK23")
    
            # Extract solution at final time  
            U_approx = sol.y[:, -1]
    
            # Exact solution at final time
            x = w * (x1 + x2) / (2 * np.pi) - x1
            x0 = w0 * (x1 + x2) / (2 * np.pi) - x1
            U_exact = u_exact(x, tf, c_value, x0)
    
            # Compute the L2 error
            err = U_approx - U_exact
            weights = np.ones_like(err) * 2 * np.pi / N
            error = discrete_L2_norm(err, weights)  # Use the discrete L2 norm
            errors.append(error)  # Append error for the current N
    
        # Plot errors for the current c_value
        ax_convergence.semilogy(N_values, errors, ".-", label=fr"$c = {c_value}$")
    
    # Customize the convergence subplot
    ax_convergence.set_xlabel(r"$N$", fontsize=label_fontsize)
    ax_convergence.set_ylabel(r"$\Vert u_N(x,T) - u(x,T) \Vert_{L^2}$", fontsize=label_fontsize)
    ax_convergence.set_title("Convergence Plot", fontsize=title_fontsize)
    ax_convergence.legend(fontsize=label_fontsize)

    plt.tight_layout()
    
    # Save the figure as a PNG file
    plt.savefig("/home/max/Documents/DTU/AdvNumericalMethods/AdvSciComp/Assignment2/figures/1d_solutions.png", format='png', bbox_inches="tight")
    plt.show()



