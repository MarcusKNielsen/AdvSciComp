import numpy as np

def discrete_inner_product(u,v,w):    
    return np.sum(u*v*w)

def discrete_L2_norm(u,w):
    return np.sqrt(discrete_inner_product(u,u,w))

def compute_L2_error(numerical, exact, weights):
    return np.sqrt(discrete_inner_product(numerical - exact, numerical - exact, weights))
