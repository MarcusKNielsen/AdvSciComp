def convergence_list_poly(N_list,poly_approx,u_func,uk_func,k_lin=np.arange(200)):

    trunc_err = []

    for N in N_list:

        x_lin = np.linspace(0,2*np.pi,N)

        u_approx = poly_approx(len(k_lin),x_lin,uk_func(k_lin,N))

        trunc_err.append(np.max(np.abs((u_func(x_lin)-u_approx))))

    return trunc_err



def discrete_poly_coefficients(k_lin,N,alpha=0,beta=0,u_func=lambda x: 1/(2-np.cos(x))):

    uk_approx   = np.zeros_like(k_lin,dtype=complex)
        
    for k_idx,k in enumerate(k_lin): 
        s = 0
        yk = 0
        for j in range(N):
            xj = 2*np.pi*j/N # Ikke rigtige skal ændres !!
            wj = (1-xj)**alpha+(1+xj)**beta
            phi_k = JacobiP(xj,alpha=0,beta=0,N=j)
            s += (u_func(xj))*phi_k*wj
            yk += phi_k**2*wj 

        uk_approx[k_idx] = s/yk

    return uk_approx