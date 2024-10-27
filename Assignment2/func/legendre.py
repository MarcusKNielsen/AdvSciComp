import numpy as np
from JacobiGL import JacobiGL

def poly(x,n,Normalize=True):
    
    n = n + 1
    
    is_int   = isinstance(x, int)
    is_float = isinstance(x, float)

    if is_int or is_float:
        x = np.array([x])

    # Legendre polynomials
    Lpre = np.ones_like(x)
    
    norm = 1
    
    if n == 0:
        if Normalize:
            norm = np.sqrt(2)
        return Lpre/norm
    
    Lnow = x
    
    if n == 1:
        if Normalize:
            norm = np.sqrt(2/3)
        return Lnow/norm

    for k in range(1,n-1):
        
        # Legendre polynomial and derivative
        Lnxt  = ((2*k+1)*x*Lnow - k*Lpre )/(k+1)
        
        if Normalize:
            norm = np.sqrt(2/(2*(k+1)+1))

        Lpre = Lnow
        Lnow = Lnxt

    return Lnxt/norm

def vander(x,N=None,Normalize=True):
    """
    Normalization is continuous L2,
    for discrete L2 normalization see page 35
    equation 1.135 and 1.136 Kopriva
    """
    
    M = len(x)
    
    if N == None:
        N = M
    
    # Init matrices
    V  = np.zeros([M,N])
    Vx = np.zeros([M,N])
    
    # Legendre polynomials
    Lpre = np.ones_like(x)
    Lnow = x
    
    # derivative of Legendre polynomials
    Lxpre = np.zeros_like(x)
    Lxnow = np.ones_like(x)
    
    # if Normalize == None then norm = 1 is used
    norm = 1
    
    if Normalize:
        norm = np.sqrt(2)
        
    V[:,0]  = Lpre  / norm
    Vx[:,0] = Lxpre / norm
    
    if Normalize:
        norm = np.sqrt(2/3)
    
    V[:,1]  = Lnow  / norm
    Vx[:,1] = Lxnow / norm
    
    for k in range(1,N-1):
        
        # Legendre polynomial and derivative
        Lnxt  = ((2*k+1)*x*Lnow - k*Lpre )/(k+1)
        Lxnxt =  (2*k+1)*Lnow+Lxpre
        
        if Normalize:
            if k+1 < N-1:
                norm = np.sqrt(2/(2*(k+1)+1))
            else:
                norm = np.sqrt(2/(k+1))
                
        V[:,k+1]  = Lnxt / norm
        Vx[:,k+1] = Lxnxt / norm
        Lpre = Lnow
        Lnow = Lnxt
        Lxpre = Lxnow
        Lxnow = Lxnxt
    
    w = 2/((N-1)*N) * 1/Lnow**2
        
    return V,Vx,w


def nodes(N):
    x = JacobiGL(0,0,N-1)
    return x






