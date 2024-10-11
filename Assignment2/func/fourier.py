import numpy as np

def nodes(N):
    j = np.arange(N)
    x = 2*np.pi*j/N
    return x

def diff_matrix(N):
    
    D = np.zeros([N,N])
    for i in range(N):
        for j in range(N):
            if i == j:
                D[j,i] = 0
            else:
                if N % 2 == 0:
                    D[j,i] = 0.5*(-1)**(i+j)/np.tan(np.pi*(j-i)/N)
                else:
                    D[j,i] = 0.5*(-1)**(i+j)/np.sin(np.pi*(j-i)/N)
    return D

if __name__ == "__main__":

    N = 16
    x = nodes(N)
    D = diff_matrix(N)

    print(np.allclose(np.sum(D,axis=1),np.zeros(N)))

    
    
    
    
    
    
    
    