
import numpy as np
from scipy.sparse import diags

def divergence(Y, X):
    return np.add(np.gradient(Y, axis=0), np.gradient(X, axis=1))

def getLaplacianMatrix(n):
    # generate [-2,-3,-2,-3,-4,-3,-2,-3,-2]
    D = -4*np.ones((n,n), dtype=np.int)
    D[0]=-3*np.ones((1,n), dtype=np.int)
    D[n-1]=-3*np.ones((1,n), dtype=np.int)
    D[:,0]=-3*np.ones(n, dtype=np.int)
    D[:,n-1]=-3*np.ones(n, dtype=np.int)
    D[0,0]=-2; D[0,n-1]=-2; D[n-1,0]=-2; D[n-1,n-1]=-2
    D = D.flatten()

    # generate [1,1,0,1,1,0,1,1]
    D1 = np.ones((n,n), dtype=np.int)
    D1[0] = np.zeros(n, dtype=np.int)
    D1 = np.transpose(D1)
    D1 = D1.flatten()[1:]

    # generate [1,1,1,1,1,1]
    D2 = np.ones(n*(n-1), dtype=int)

    # return matrix
    diagonals = [D2, D1, D, D1, D2]
    return diags(diagonals, [-n, -1, 0, 1, n])

