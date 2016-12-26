
import numpy as np
from scipy.sparse import identity
from scipy.sparse.linalg import minres

def diffuse (S, L, eta, dt):
    n, _ = S.shape
    n1 = n-1
    n2 = n1-1
    S = S[1:n1, 1:n1]
    L = identity(n2*n2, format='dia') - L.multiply(eta*dt)
    S = S.flatten()
    s2,_ = minres(L, S, S)
    s2 = np.reshape(s2, (n2,n2))
    S2 = np.zeros((n,n), dtype=np.float)
    S2[1:n1, 1:n1] = s2
    return S2
