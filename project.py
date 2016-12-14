
# coding: utf-8

# In[10]:

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt

def divergence(X, Y):
    return np.add(np.gradient(X, axis=1), np.gradient(Y, axis=0))

def getPoissonMatrix(n):
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

def pressure_solve(U, V, M):
    n,_ = U.shape
    n1 = n-1
    n2 = n-2
    n3 = n-3

    #plot1 = plt.figure()
    #plt.quiver(U, V)
    #plt.title("Before Projection")
    #plt.show(plot1)

    W = divergence(U,V)
    W = W[1:n1, 1:n1]
    W = W.flatten()

    x, _ = cg(M, W)
    x = np.reshape(x,(n2,n2))
    X = np.zeros((n,n), dtype = np.float)
    X[1:n1, 1:n1] = x
    X[0, 1:n1] = x[0]
    X[n1, 1:n1] = x[n3]
    X[1:n1, 0] = x[:,0]
    X[1:n1, n1] = x[:,n3]
    X[0,0] = x[0,0]
    X[0,n1] = x[0,n3]
    X[n1,0] = x[n3,0]
    X[n1,n1] = x[n3,n3]

    U = U - np.gradient(X, axis=1)
    V = V - np.gradient(X, axis=0)

    return (U,V,X)


# In[ ]:



