import numpy as np
from boundaries import set_bounds

def interpolate(x, i, j):
    d1, d2 = x.shape

    if (j<0):
        j = j-j//1
    if (i<0):
        i = i-i//1
    if (j>=d2-1):
        j = d2-2+j%1
    if (i>=d1-1):
        i = d1-2+i%1

    j0 = j//1; dj = j%1
    i0 = i//1; di = i%1

    d00 = x.item(i0,j0)
    d01 = x.item(i0,j0+1)
    d10 = x.item(i0+1,j0)
    d11 = x.item(i0+1,j0+1)

    return (1-dj)*(1-di)*d00 + dj*(1-di)*d01 + (1-dj)*di*d10 + dj*di*d11

def advect(x,x2,u,v,dt,b):
    d1,d2 = x.shape
    for i in range(1,d1-1):
        for j in range(1,d2-1):
            i2 = i-v[i,j]*dt # trace back
            j2 = j-u[i,j]*dt # trace back
            x2[i,j] = interpolate(x,i2,j2)
    set_bounds(x2,b)
    return x2,x
