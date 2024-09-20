import numpy as np
from scipy.special import gamma

def JacobiGQ(alpha, beta, N):
    """
    Compute the N'th order Gauss quadrature points, x, 
    and weights, w, associated with the Jacobi polynomial of type (alpha, beta) > -1 (<> -0.5).
    
    Parameters:
    alpha (float): First parameter of the Jacobi polynomial.
    beta (float): Second parameter of the Jacobi polynomial.
    N (int): The order of the quadrature.
    
    Returns:
    x (np.ndarray): The quadrature points.
    w (np.ndarray): The quadrature weights.
    """

    # Special case for N = 0
    if N == 0:
        x = np.array([-(alpha - beta) / (alpha + beta + 2)])
        w = np.array([2])
        return x, w

    # Form symmetric matrix from recurrence.
    J = np.zeros((N + 1, N + 1))
    h1 = 2 * np.arange(N + 1) + alpha + beta

    J = np.diag(-0.5 * (alpha**2 - beta**2) / ((h1 + 2) * h1)) + \
        np.diag(2 / (h1[:N] + 2) * np.sqrt((np.arange(1, N + 1) * 
             (np.arange(1, N + 1) + alpha + beta) *
             (np.arange(1, N + 1) + alpha) *
             (np.arange(1, N + 1) + beta) / 
             ((h1[:N] + 1) * (h1[:N] + 3)))), 1)

    if alpha + beta < 10 * np.finfo(float).eps:
        J[0, 0] = 0.0

    J = J + J.T

    # Compute quadrature by eigenvalue solve
    D, V = np.linalg.eig(J)
    x = np.sort(D)
    w = (V[0, :]**2) * 2**(alpha + beta + 1) / (alpha + beta + 1) * \
        gamma(alpha + 1) * gamma(beta + 1) / gamma(alpha + beta + 1)

    return x, w
