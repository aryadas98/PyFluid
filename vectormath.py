import numpy as np

# calculates gradient, taking boundaries into account
# uses central differences for points not surrounded by boundaries
# uses left or right differences for other points

def gradient(X, y_axis, res, b, out):
    d1,d2 = X.shape

    if y_axis:
        for i in range(1,d1-1):
            for j in range(1,d2-1):
                if b[i,j]:
                    if (b[i-1,j] and b[i+1,j]):
                        out[i,j] = (X[i+1,j]-X[i-1,j])*res/2
                    elif b[i-1,j]:
                        out[i,j] = (X[i,j]-X[i-1,j])*res
                    elif b[i+1,j]:
                        out[i,j] = (X[i+1,j]-X[i,j])*res
    else:
        for i in range(1,d1-1):
            for j in range(1,d2-1):
                if b[i,j]:
                    if (b[i,j-1] and b[i,j+1]):
                        out[i,j] = (X[i,j+1]-X[i,j-1])*res/2
                    elif b[i,j-1]:
                        out[i,j] = (X[i,j]-X[i,j-1])*res
                    elif b[i,j+1]:
                        out[i,j] = (X[i,j+1]-X[i,j])*res
    return out



# calculates divergence, taking boundaries into account
# uses central differences for points not surrounded by boundaries
# uses left or right differences for other points

def divergence(X, Y, b, res, out):
    d1,d2 = X.shape
    out.fill(0)

    for i in range(1,d1-1):
        for j in range(1,d2-1):
            if b[i,j]:
                if (b[i-1,j] and b[i+1,j]):
                    out[i,j] += (Y[i+1,j]-Y[i-1,j])/2/res
                elif b[i-1,j]:
                    out[i,j] += (Y[i,j]-Y[i-1,j])/res
                elif b[i+1,j]:
                    out[i,j] += (Y[i+1,j]-Y[i,j])/res

                if (b[i,j-1] and b[i,j+1]):
                    out[i,j] += (X[i,j+1]-X[i,j-1])/2/res
                elif b[i,j-1]:
                    out[i,j] += (X[i,j]-X[i,j-1])/res
                elif b[i,j+1]:
                    out[i,j] += (X[i,j+1]-X[i,j])/res
    return out

