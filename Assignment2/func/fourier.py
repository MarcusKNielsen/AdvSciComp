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
 
def diff_matrix2(N):

    j_lin = np.arange(N)
    xj = 2*np.pi*j_lin/N
    Dh = lambda xj,x,N: np.where(
        np.abs(x-xj) < 1e-12,
        0,
        (np.cos(N*(x - xj)/2)*np.cos(x/2 - xj/2)*N*np.sin(x/2 - xj/2) - np.sin(N*(x - xj)/2))/(2*np.sin(x/2 - xj/2)**2*N)
    )

    D = np.zeros([N,N])
    for j in j_lin:
        D[:,j] = Dh(xj[j],xj,N)
    
    return D#,Dh


if __name__ == "__main__":

    N = 16
    #x = nodes(N)

    D = diff_matrix(N)

    print(np.allclose(np.sum(D,axis=1),np.zeros(N)))

    
    
    
    
    
    
    
    