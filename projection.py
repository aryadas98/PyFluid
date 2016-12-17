
import numpy as np
from scipy.sparse.linalg import cg
from vectormath import divergence

def pressure_solve(V, U, M, x0):
    n,_ = U.shape
    n1 = n-1
    n2 = n-2
    n3 = n-3

    #plot1 = plt.figure()
    #plt.quiver(U, V)
    #plt.title("Before Projection")
    #plt.show(plot1)

    W = divergence(V,U)
    W = W[1:n1, 1:n1]
    W = W.flatten()

    x0 = x0[1:n1, 1:n1].flatten()

    x, _ = cg(M, W, x0, 0.4)
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

    return (V,U,X)

