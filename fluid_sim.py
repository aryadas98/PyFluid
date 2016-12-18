
from vectormath import divergence, getLaplacianMatrix
from projection import pressure_solve
from advection import advect
from diffusion import diffuse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

n = 25

U = np.zeros((n,n), dtype=np.float)
V = np.zeros((n,n), dtype=np.float)
P = np.zeros((n,n), dtype=np.float)
S = np.zeros((n,n), dtype=np.float)

M = getLaplacianMatrix(n-2)

dt = 0.1
smoke_dt = 5*dt
eta =0.1

fig = plt.figure()
plt.axis('off')
plot = plt.imshow(S, cmap='gray', origin='lower', vmin=0, vmax=3, interpolation='bicubic')

def init_anim():
    global U,V,S,plot

    # At the moment, the initial conditions have to be set manually.
    # I have provided two sets of initial conditions. Use any one.
    # To use any one initial condition, uncomment it, and comment others.

    # Set 1:

    # U[n//2-7:n//2-4, n//2-7:n//2-4] = -10*np.ones((3,3), dtype=float)
    # U[n//2+4:n//2+7, n//2+4:n//2+7] = 10*np.ones((3,3), dtype=float)
    # S[n//2-7:n//2-4, n//2-7:n//2-4] = 10*np.ones((3,3), dtype=float)
    # S[n//2+4:n//2+7, n//2+4:n//2+7] = 10*np.ones((3,3), dtype=float)

    # Set 2:
    
    U[n//2-10:n//2-7, n//2-1:n//2+2] = -10*np.ones((3,3), dtype=float)
    U[n//2+7:n//2+10, n//2-1:n//2+2] = 10*np.ones((3,3), dtype=float)
    V[n//2-1:n//2+2, n//2-10:n//2-7] = 10*np.ones((3,3), dtype=float)
    V[n//2-1:n//2+2, n//2+7:n//2+10] = -10*np.ones((3,3), dtype=float)
    S[n//2-1:n//2+2, n//2+7:n//2+10] = 10*np.ones((3,3), dtype=float)
    S[n//2-1:n//2+2, n//2-10:n//2-7] = 10*np.ones((3,3), dtype=float)
    S[n//2-10:n//2-7, n//2-1:n//2+2] = 10*np.ones((3,3), dtype=float)
    S[n//2+7:n//2+10, n//2-1:n//2+2] = 10*np.ones((3,3), dtype=float)

    plot.set_data(S)
    return [plot]


def animate(i):
    global U,V,S,P,plot

    S = diffuse(S, M, eta, dt)
    U = diffuse(U, M, eta, dt)
    V = diffuse(V, M, eta, dt)

    V, U, P = pressure_solve(V, U, M, P)

    U = advect(V, U, U, dt)
    V = advect(V, U, V, dt)
    S = advect(V, U, S, smoke_dt)

    V, U, P = pressure_solve(V, U, M, P)

    plot.set_data(S)
    return [plot]

anim = animation.FuncAnimation(fig, animate, init_func=init_anim, frames=70, interval=5, blit=True)

# Save the animation. If you want to save the animation, uncomment the line below.
# anim.save('animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
