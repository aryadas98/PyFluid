import numpy as np
from vectormath import gradient,divergence
from boundaries import set_bounds

def project(u,v,res,div,p,px,py,b,b2):
    d1,d2 = u.shape
    divergence(u,v,b,res,div)
    # zero pressure to keep things stable
    p.fill(0)
    set_bounds(p,b)

    # Gauss-Seidel solver
    for _ in range(20):
        for i in range(1,d1-1):
            for j in range(1,d2-1):
                if b[i,j]:
                    p[i,j] = (p[i,j-1]+p[i,j+1]+p[i-1,j]+p[i+1,j]-div[i,j])/b2[i,j]
        set_bounds(p,b)


    u -= gradient(p,False,res,b,px)
    v -= gradient(p,True,res,b,py)

    set_bounds(u,b)
    set_bounds(v,b)
