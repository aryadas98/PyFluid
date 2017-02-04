import numpy as np
import scipy.ndimage.filters as filters

import matplotlib.pyplot as plt
from matplotlib import animation

from diffusion import diffuse
from advection import advect
from projection import project

def init_anim():
    global res,d1,d2, \
        u,u2,v,v2,s,s2,div,p,px,py, \
        dt2,a,b,b2,plot,plot2

    # simulator settings
    res = 30
    shape = (1*res,3*res)
    d1,d2 = shape

    u,u2,v,v2,s,s2,div,p,px,py = (np.zeros(shape) for _ in range(10))

    dt = 0.05; visc = 1e-6
    a = visc*dt*res**2
    dt2 = dt*res

    # scenario

    # boundaries
    b = np.ones(shape)
    b[0] = 0; b[-1] = 0
    b[d1//2-2:d1//2+3,d1//2-2:d1//2+3] = 0
    b2 = filters.convolve(b,[[0,1,0],[1,0,1],[0,1,0]])

    # sources and sinks
    u[:,0:2] = 2
    s[:,0:2] = 0
    s[::3,0:2] = 1

    # animation
    plot = plt.imshow(s,cmap='gray',interpolation='bicubic',vmin=0,vmax=1,origin='lower')
    plot2 = plt.imshow(1-b,cmap='Blues',alpha=0.1,interpolation='nearest',origin='lower')
    return [plot]

def animate(i):
    global res,d1,d2, \
        u,u2,v,v2,s,s2,div,p,px,py, \
        dt2,a,b,b2,plot,plot2

    # add forces
    u[:,0:2] = 2
    s[:,0:2] = 0
    s[::3,0:2] = 1
    s[:,-1:-3] = 0

    # diffuse
    u,u2 = diffuse(u,u2,a,b)
    v,v2 = diffuse(v,v2,a,b)
    s,s2 = diffuse(s,s2,a,b)

    # advect
    project(u,v,res,div,p,px,py,b,b2)

    s,s2 = advect(s,s2,u,v,dt2,b)
    u,u2 = advect(u,u2,u,v,dt2,b)
    v,v2 = advect(v,v2,u2,v,dt2,b)

    # project
    project(u,v,res,div,p,px,py,b,b2)

    plot.set_data(s)
    return [plot]

fig = plt.figure()
plt.axis('off')
anim = animation.FuncAnimation(fig,animate,init_func=init_anim,frames=600,blit=True)
# uncomment the line below to save the animation
#anim.save('animation.mp4', fps=20, extra_args=['-vcodec','libx264'])
plt.show()
