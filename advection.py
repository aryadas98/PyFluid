
import numpy as np

def interpolate(M, y, x):
    n, _ = M.shape

    if (x<0):
        x = x-x//1
    if (y<0):
        y = y-y//1
    if (x>=n-1):
        x = n-2+x%1
    if (y>=n-1):
        y = n-2+y%1

    x0 = x//1; dx = x%1
    y0 = y//1; dy = y%1

    d00 = M.item(y0,x0)
    d01 = M.item(y0,x0+1)
    d10 = M.item(y0+1,x0)
    d11 = M.item(y0+1,x0+1)

    return (1-dx)*(1-dy)*d00 + dx*(1-dy)*d01 + (1-dx)*dy*d10 + dx*dy*d11

def advect(V, U, S, dt):
    n, _ = U.shape
    x = np.tile(np.arange(n), (n,1))
    y = np.transpose(x)

    x = x - dt*U
    y = y - dt*V

    S2 = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            S2.itemset((i,j), interpolate(S, y.item(i,j), x.item(i,j)))

    return S2

