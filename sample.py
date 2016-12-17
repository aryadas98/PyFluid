
from vectormath import divergence, getLaplacianMatrix
from projection import pressure_solve
from advection import advect
from diffusion import diffuse
import numpy as np
import matplotlib.pyplot as plt

n = 25
U = np.zeros((n,n), dtype=np.float)
V = np.zeros((n,n), dtype=np.float)
P = np.zeros((n,n), dtype=np.float)
S = np.zeros((n,n), dtype=np.float)

dt = 0.1
eta =0.1

#V[n//2-5:n//2+5, 0:n] = 1*np.ones((10,n), dtype=float)
#U.itemset((n//2,n//2+1), 5)
#U.itemset((n//2,n//2), 5)
#U.itemset((n//2,n//2-1), 5)

U[n//2-10:n//2-7, n//2-1:n//2+2] = 8*np.ones((3,3), dtype=float)
S[n//2-10:n//2-7, n//2-5:n//2+2] = 3*np.ones((3,1), dtype=float)
#V[n//2-1:n//2+2, n//2-1:n//2+2] = np.ones((3,3), dtype=float)


#U.itemset((n//2+5,n//2-5), -10)
#U.itemset((n//2+5,n//2-4), -10)
#U.itemset((n//2+5,n//2-6), -10)

M = getLaplacianMatrix(n-2)
#M2 = getLaplacianMatrix(n)

#V, U, P = pressure_solve(V, U, M, P)
i=0

while True:
    #speed = np.sqrt(U**2 + V**2)
    #UN = U/speed
    #VN = V/speed

    S = diffuse(S, M, eta, dt)
    U = diffuse(U, M, eta, dt)
    V = diffuse(V, M, eta, dt)

    #plot2 = plt.figure()
    #plt.title("After Diffusion")
    #plt.quiver(U, V)
    #plt.show(plot2)

    V, U, P = pressure_solve(V, U, M, P)

    #plot2 = plt.figure()
    #plt.title("After Projection")
    #plt.quiver(U,V)
    #plt.show(plot2)
    #plot2.savefig('anim/frame'+str(i)+'.png')
    #print('frame'+str(i)+' printed.')

    plt.imshow(S, cmap='gray', origin='lower')
    #plt.show(plot3)
    plt.savefig('anim2/frame'+str(i)+'.png')
    print('frame'+str(i)+' printed.')

    U = advect(V, U, U, dt)
    V = advect(V, U, V, dt)
    S = advect(V, U, S, dt)

    #plot2 = plt.figure()
    #plt.title("After Advection")
    #plt.quiver(U,V)
    #plt.show()

    V, U, P = pressure_solve(V, U, M, P)
    #print(np.sum(np.abs(divergence(V,U))))
    #print(d)
    #plot2 = plt.figure()
    #plt.title("After Projection")
    #plt.quiver(U, V)
    #plt.show(plot2)
    i=i+1
