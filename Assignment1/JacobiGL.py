import numpy as np
from JacobiGQ import JacobiGQ

def JacobiGL(alpha, beta, N):
    """
    Compute the N'th order Gauss-Lobatto quadrature points, x, 
    associated with the Jacobi polynomial of type (alpha, beta) > -1 (<> -0.5).
    
    Parameters:
    alpha (float): First parameter of the Jacobi polynomial.
    beta (float): Second parameter of the Jacobi polynomial.
    N (int): The order of the quadrature.
    
    Returns:
    x (np.ndarray): The Gauss-Lobatto quadrature points.
    """
    
    x = np.zeros(N + 1)
    
    # Special case when N = 1
    if N == 1:
        x[0] = -1.0
        x[1] = 1.0
        return x

    # Compute internal Gauss quadrature points
    xint, _ = JacobiGQ(alpha + 1, beta + 1, N - 2)

    # Combine endpoints with internal points
    x = np.concatenate(([-1], xint, [1]))

    return x
