import numpy as np
from scipy.special import gamma

"""
See legendre.py if legendre is needed
"""

def vander(x,alpha=0,beta=0,N=None):
    
    M = len(x)
    
    if N == None:
        N = M
    
    V  = np.zeros([M,N])
    
    J_nm2 = np.ones_like(x)
    J_nm1 = 1/2*(alpha-beta+(alpha+beta+2)*x)

    V[:,0] = J_nm2 / np.sqrt(2**(alpha+beta+1)*gamma(alpha+1)*gamma(beta+1)/((alpha+beta+1)*gamma(alpha+beta+1)))
    V[:,1] = J_nm1 / np.sqrt(2**(alpha+beta+1)*gamma(1+alpha+1)*gamma(1+beta+1)/(gamma(2)*(2+alpha+beta+1)*gamma(alpha+beta+1)))

    for n in range(1,N-1):

        # Computing the recursive coefficients
        anm2  = 2*(n+alpha)*(n+beta)/( (2*n+alpha+beta+1)*(2*n+alpha+beta) )
        anm1  = (alpha**2-beta**2)/( (2*n+alpha+beta+2)*(2*n+alpha+beta) )
        an    = 2*(n+1)*(n+beta+alpha+1)/( (2*n+alpha+beta+2)*(2*n+alpha+beta+1) )
        
        # Computing
        J_n = ( (anm1 + x )*J_nm1 - anm2*J_nm2 ) / an

        norm = np.sqrt(2**(alpha+beta+1)*gamma((n+1)+alpha+1)*gamma((n+1)+beta+1)/(gamma((n+1)+1)*(2*(n+1)+alpha+beta+1)*gamma((n+1)+alpha+beta+1)))

        # Normalize and insert in V array
        V[:,n+1] = J_n / norm
        
        # Updating step
        J_nm2 = J_nm1
        J_nm1 = J_n
    
    return V




