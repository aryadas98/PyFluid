import numpy as np
from boundaries import set_bounds

def diffuse(x,x2,a,b):
    d1,d2 = x.shape

    # Gauss-Seidel solver
    for _ in range(20):
        for i in range(1,d1-1):
            for j in range(1,d2-1):
                if b[i,j]:
                    x2[i,j] = (x[i,j]+a*(x2[i,j-1]+x2[i,j+1]+x2[i-1,j]+x2[i+1,j]))/(1+4*a)

    set_bounds(x2,b)
    return x2,x
