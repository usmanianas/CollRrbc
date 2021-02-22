
import numpy as np
import pylab as plt
from matplotlib import cm
import h5py as hp
import time
import random 



Lx, Ly, Lz = 1.0, 1.0, 1.0

L, M, N = 25, 25, 25

Nx, Ny, Nz = L, M, N

hx, hy, hz = Lx/(L-1), Ly/(M-1), Lz/(N-1)

x = np.linspace(0, 1, Nx, endpoint=True)        
y = np.linspace(0, 1, Ny, endpoint=True)
z = np.linspace(0, 1, Nz, endpoint=True)    

hx2, hy2, hz2 = hx*hx, hy*hy, hz*hz

hx2hy2, hy2hz2, hz2hx2 = hx2*hy2, hy2*hz2, hz2*hx2

hx2hy2hz2 = hx2*hy2*hz2

idx2, idy2, idz2 = 1.0/hx2, 1.0/hy2, 1.0/hz2

Ra = 1.0e4

Pr = 1

Ta = 0.0e7

print("#", "Ra=", Ra, "Pr=", Pr, "Ta=", Ta)

#Ro = np.sqrt(Ra/(Ta*Pr))

nu, kappa = np.sqrt(Pr/Ra), 1.0/np.sqrt(Ra*Pr)

#print(nu, kappa)

dt = 0.02

tMax = 1000

restart = 0    # 0-Fresh, 1-Restart

# Number of iterations after which output must be printed to standard I/O
opInt = 1

# File writing interval
fwInt = 2

# Tolerance value in Jacobi iterations
VpTolerance = 1.0e-5

# Tolerance value in Jacobi iterations
PoissonTolerance = 1.0e-3

gssor = 1.6

maxCount = 1e4




def getDiv(U, V, W):

    divMat = np.zeros([L, M, N])

    divMat[1:L-1, 1:M-1, 1:N-1] = ((U[2:L, 1:M-1, 1:N-1] - U[0:L-2, 1:M-1, 1:N-1])*0.5/hx +
                                (V[1:L-1, 2:M, 1:N-1] - V[1:L-1, 0:M-2, 1:N-1])*0.5/hy +
                                (W[1:L-1, 1:M-1, 2:N] - W[1:L-1, 1:M-1, 0:N-2])*0.5/hz)
    
    #return np.unravel_index(divMat.argmax(), divMat.shape), np.mean(divMat)
    return np.max(divMat)
    



def writeSoln(U, V, W, P, T, time):

    fName = "Soln_" + "{0:09.5f}.h5".format(time)
    print("#Writing solution file: ", fName)        
    f = hp.File(fName, "w")

    dset = f.create_dataset("U", data = U)
    dset = f.create_dataset("V", data = V)
    dset = f.create_dataset("W", data = W)
    dset = f.create_dataset("T", data = T)
    dset = f.create_dataset("P", data = P)

    f.close()



def initFields():
    global L, M, N, Nx, Ny, Nz
    global U, V, W, P, T
    global Hx, Hy, Hz, Ht, Pp
    global restart

    P = np.ones([L, M, N])

    #T = random.uniform(0, 1) * np.ones([L, M, N])
    T = np.zeros([L, M, N])

    T[:, :, 0:N] = 1 - z[0:N]

    U = np.zeros([L, M, N])

    #V = random.uniform(0, 1) * np.ones([L, M, N])
    V = np.zeros([L, M, N])

    #W = random.uniform(0, 1) * np.ones([L, M, N])
    W = np.zeros([L, M, N])

    time = 0

    filename = "Soln_500.00000.h5"

    if restart == 1:
        def hdf5_reader(filename,dataset):
            file_V1_read = hp.File(filename)
            dataset_V1_read = file_V1_read["/"+dataset]
            V1=dataset_V1_read[:,:,:]
            return V1
    
        U = hdf5_reader(filename, "U")
        V = hdf5_reader(filename, "V")
        W = hdf5_reader(filename, "W")
        P = hdf5_reader(filename, "P")
        T = hdf5_reader(filename, "T")

    writeSoln(U, V, W, P, T, time)

    #print(np.amax(U), np.amax(V), np.amax(W))

    # Define arrays for storing RHS of NSE
    Hx = np.zeros_like(U)
    Hy = np.zeros_like(V)
    Hz = np.zeros_like(W)
    Ht = np.zeros_like(T)   
    Pp = np.zeros_like(P)

    #if probType == 0:
        # For moving top lid, U = 1.0 on lid, and second last point lies on the wall
    #    U[:, :, -2] = 1.0
    #if probType == 1:
        # Initial condition for forced channel flow
    #    U[:, :, :] = 1.0

    #ps.initVariables()

    #if testPoisson:
    #    ps.initDirichlet()


def TimeIntegrate():

    global N, M, L, Nx, Ny, Nz, hx, hy, hz, x, y, z, dt
    global U, V, W, P, T
    global Hx, Hy, Hz, Pp, Ht
    
    time = 0
    fwTime = 0.0
    iCnt = 1
    
    Hx.fill(0.0)
    Hy.fill(0.0)
    Hz.fill(0.0)
    Ht.fill(0.0)

    while True:

        if iCnt % opInt == 0:

            Re = np.mean(np.sqrt(U[1:L-1, 1:M-1, 1:N-1]**2.0 + V[1:L-1, 1:M-1, 1:N-1]**2.0 + W[1:L-1, 1:M-1, 1:N-1]**2.0))/nu
            Nu = 1.0 + np.mean(W[1:L-1, 1:M-1, 1:N-1]*T[1:L-1, 1:M-1, 1:N-1])/kappa
            maxDiv = getDiv(U, V, W)

            print("%f    %f    %f    %f" %(time, Re, Nu, maxDiv))           


    
        Hx = computeNLinDiff_X(U, V, W)
        Hy = computeNLinDiff_Y(U, V, W)
        Hz = computeNLinDiff_Z(U, V, W)
        Ht = computeNLinDiff_T(U, V, W, T)  
    
    
    
    
        # Calculating guessed values of U implicitly
        Hx[1:L-1, 1:M-1, 1:N-1] = U[1:L-1, 1:M-1, 1:N-1] + dt*(Hx[1:L-1, 1:M-1, 1:N-1] - np.sqrt((Ta*Pr)/Ra)*(-V[1:L-1, 1:M-1, 1:N-1]) - (P[2:L, 1:M-1, 1:N-1] - P[0:L-2, 1:M-1, 1:N-1])/(2.0*hx))
        uJacobi(Hx)
    
        # Calculating guessed values of V implicitly
        Hy[1:L-1, 1:M-1, 1:N-1] = V[1:L-1, 1:M-1, 1:N-1] + dt*(Hy[1:L-1, 1:M-1, 1:N-1] - np.sqrt((Ta*Pr)/Ra)*(U[1:L-1, 1:M-1, 1:N-1]) - (P[1:L-1, 2:M, 1:N-1] - P[1:L-1, 0:M-2, 1:N-1])/(2.0*hy))
        vJacobi(Hy)
    
        # Calculating guessed values of W implicitly
        Hz[1:L-1, 1:M-1, 1:N-1] = W[1:L-1, 1:M-1, 1:N-1] + dt*(Hz[1:L-1, 1:M-1, 1:N-1] - ((P[1:L-1, 1:M-1, 2:N] - P[1:L-1, 1:M-1, 0:N-2])/(2.0*hz)) + T[1:L-1, 1:M-1, 1:N-1])
        wJacobi(Hz)
    
        # Calculating guessed values of T implicitly
        Ht[1:L-1, 1:M-1, 1:N-1] = T[1:L-1, 1:M-1, 1:N-1] + dt*Ht[1:L-1, 1:M-1, 1:N-1]
        TJacobi(Ht)   
    
        #print(np.amax(U), np.amax(V), np.amax(W))
    
        # Calculating pressure correction term
        rhs = np.zeros([L, M, N])
        rhs[1:L-1, 1:M-1, 1:N-1] = ((U[2:L, 1:M-1, 1:N-1] - U[0:L-2, 1:M-1, 1:N-1])/(2.0*hx) +
                                    (V[1:L-1, 2:M, 1:N-1] - V[1:L-1, 0:M-2, 1:N-1])/(2.0*hy) +
                                    (W[1:L-1, 1:M-1, 2:N] - W[1:L-1, 1:M-1, 0:N-2])/(2.0*hz))/dt
    
        #ps.multigrid(Pp, rhs)
    
        Pp = PoissonSolver(rhs)
    
        # Add pressure correction.
        P = P + Pp
    
        # Update new values for U, V and W
        U[1:L-1, 1:M-1, 1:N-1] = U[1:L-1, 1:M-1, 1:N-1] - dt*(Pp[2:L, 1:M-1, 1:N-1] - Pp[0:L-2, 1:M-1, 1:N-1])/(2.0*hx)
        V[1:L-1, 1:M-1, 1:N-1] = V[1:L-1, 1:M-1, 1:N-1] - dt*(Pp[1:L-1, 2:M, 1:N-1] - Pp[1:L-1, 0:M-2, 1:N-1])/(2.0*hy)
        W[1:L-1, 1:M-1, 1:N-1] = W[1:L-1, 1:M-1, 1:N-1] - dt*(Pp[1:L-1, 1:M-1, 2:N] - Pp[1:L-1, 1:M-1, 0:N-2])/(2.0*hz)

        imposeUBCs(U)                               
        imposeVBCs(V)                               
        imposeWBCs(W)                               
        imposePBCs(P)                               
        imposeTBCs(T)       

        #if abs(fwTime - time) < 0.5*dt:
        if abs(time - tMax)<1e-5:
            writeSoln(U, V, W, P, T, time)
            Z, Y = np.meshgrid(y,z)
            plt.contourf(Y, Z, T[int(Nx/2), :, :], 500, cmap=cm.coolwarm)
            clb = plt.colorbar()
            plt.quiver(Y[0:Nx,0:Ny], Z[0:Nx,0:Ny], V[int(Nx/2),:, :], W[int(Nx/2), :, :])
            plt.axis('scaled')
            clb.ax.set_title(r'$T$', fontsize = 20)
            plt.show()
            fwTime = fwTime + fwInt                                 


        if time > tMax:
            break   

    
        time = time + dt
    
        iCnt = iCnt + 1
    
        


    #print(U[30, 30, 30], V[30, 30, 30], W[30, 30, 30])


def computeNLinDiff_X(U, V, W):
    global Hx
    global N, M, L, Nx, Ny, Nz

    Hx[1:L-1, 1:M-1, 1:N-1] = (((U[2:Nx, 1:Ny-1, 1:Nz-1] - 2.0*U[1:Nx-1, 1:Ny-1, 1:Nz-1] + U[0:Nx-2, 1:Ny-1, 1:Nz-1])/hx2 + 
                                (U[1:Nx-1, 2:Ny, 1:Nz-1] - 2.0*U[1:Nx-1, 1:Ny-1, 1:Nz-1] + U[1:Nx-1, 0:Ny-2, 1:Nz-1])/hy2 + 
                                (U[1:Nx-1, 1:Ny-1, 2:Nz] - 2.0*U[1:Nx-1, 1:Ny-1, 1:Nz-1] + U[1:Nx-1, 1:Ny-1, 0:Nz-2])/hz2)*0.5*nu -
                              U[1:L-1, 1:M-1, 1:N-1]*(U[2:Nx, 1:Ny-1, 1:Nz-1] - U[0:Nx-2, 1:Ny-1, 1:Nz-1])/(2.0*hx) -
                              V[1:L-1, 1:M-1, 1:N-1]*(U[1:Nx-1, 2:Ny, 1:Nz-1] - U[1:Nx-1, 0:Ny-2, 1:Nz-1])/(2.0*hy) - 
                              W[1:L-1, 1:M-1, 1:N-1]*(U[1:Nx-1, 1:Ny-1, 2:Nz] - U[1:Nx-1, 1:Ny-1, 0:Nz-2])/(2.0*hz))

    return Hx

def computeNLinDiff_Y(U, V, W):
    global Hy
    global N, M, L, Nx, Ny, Nz

    Hy[1:L-1, 1:M-1, 1:N-1] = (((V[2:Nx, 1:Ny-1, 1:Nz-1] - 2.0*V[1:Nx-1, 1:Ny-1, 1:Nz-1] + V[0:Nx-2, 1:Ny-1, 1:Nz-1])/hx2 + 
                                (V[1:Nx-1, 2:Ny, 1:Nz-1] - 2.0*V[1:Nx-1, 1:Ny-1, 1:Nz-1] + V[1:Nx-1, 0:Ny-2, 1:Nz-1])/hy2 + 
                                (V[1:Nx-1, 1:Ny-1, 2:Nz] - 2.0*V[1:Nx-1, 1:Ny-1, 1:Nz-1] + V[1:Nx-1, 1:Ny-1, 0:Nz-2])/hz2)*0.5*nu -
                              U[1:L-1, 1:M-1, 1:N-1]*(V[2:Nx, 1:Ny-1, 1:Nz-1] - V[0:Nx-2, 1:Ny-1, 1:Nz-1])/(2.0*hx) -
                              V[1:L-1, 1:M-1, 1:N-1]*(V[1:Nx-1, 2:Ny, 1:Nz-1] - V[1:Nx-1, 0:Ny-2, 1:Nz-1])/(2.0*hy) - 
                              W[1:L-1, 1:M-1, 1:N-1]*(V[1:Nx-1, 1:Ny-1, 2:Nz] - V[1:Nx-1, 1:Ny-1, 0:Nz-2])/(2.0*hz))

    return Hy


def computeNLinDiff_Z(U, V, W):
    global Hz
    global N, M, L, Nx, Ny, Nz

    Hz[1:L-1, 1:M-1, 1:N-1] = (((W[2:Nx, 1:Ny-1, 1:Nz-1] - 2.0*W[1:Nx-1, 1:Ny-1, 1:Nz-1] + W[0:Nx-2, 1:Ny-1, 1:Nz-1])/hx2 + 
                                (W[1:Nx-1, 2:Ny, 1:Nz-1] - 2.0*W[1:Nx-1, 1:Ny-1, 1:Nz-1] + W[1:Nx-1, 0:Ny-2, 1:Nz-1])/hy2 + 
                                (W[1:Nx-1, 1:Ny-1, 2:Nz] - 2.0*W[1:Nx-1, 1:Ny-1, 1:Nz-1] + W[1:Nx-1, 1:Ny-1, 0:Nz-2])/hz2)*0.5*nu -
                              U[1:L-1, 1:M-1, 1:N-1]*(W[2:Nx, 1:Ny-1, 1:Nz-1] - W[0:Nx-2, 1:Ny-1, 1:Nz-1])/(2.0*hx) -
                              V[1:L-1, 1:M-1, 1:N-1]*(W[1:Nx-1, 2:Ny, 1:Nz-1] - W[1:Nx-1, 0:Ny-2, 1:Nz-1])/(2.0*hy) - 
                              W[1:L-1, 1:M-1, 1:N-1]*(W[1:Nx-1, 1:Ny-1, 2:Nz] - W[1:Nx-1, 1:Ny-1, 0:Nz-2])/(2.0*hz))


    return Hz


def computeNLinDiff_T(U, V, W, T):
    global Ht
    global N, M, L

    Ht[1:L-1, 1:M-1, 1:N-1] = (((T[2:Nx, 1:Ny-1, 1:Nz-1] - 2.0*T[1:Nx-1, 1:Ny-1, 1:Nz-1] + T[0:Nx-2, 1:Ny-1, 1:Nz-1])/hx2 + 
                                (T[1:Nx-1, 2:Ny, 1:Nz-1] - 2.0*T[1:Nx-1, 1:Ny-1, 1:Nz-1] + T[1:Nx-1, 0:Ny-2, 1:Nz-1])/hy2 + 
                                (T[1:Nx-1, 1:Ny-1, 2:Nz] - 2.0*T[1:Nx-1, 1:Ny-1, 1:Nz-1] + T[1:Nx-1, 1:Ny-1, 0:Nz-2])/hz2)*0.5*kappa -
                              U[1:L-1, 1:M-1, 1:N-1]*(T[2:Nx, 1:Ny-1, 1:Nz-1] - T[0:Nx-2, 1:Ny-1, 1:Nz-1])/(2.0*hx)-
                              V[1:L-1, 1:M-1, 1:N-1]*(T[1:Nx-1, 2:Ny, 1:Nz-1] - T[1:Nx-1, 0:Ny-2, 1:Nz-1])/(2.0*hy) - 
                              W[1:L-1, 1:M-1, 1:N-1]*(T[1:Nx-1, 1:Ny-1, 2:Nz] - T[1:Nx-1, 1:Ny-1, 0:Nz-2])/(2.0*hz))

    return Ht


#Jacobi iterative solver for U
def uJacobi(rho):
    global hx2, hy2, hz2, hy2hz2, hz2hx2, hx2hy2, hx2hy2hz2, nu, dt, VpTolerance, maxCount
    global U
    global L, M, N, Nx, Ny, Nz

    temp_sol = np.zeros_like(rho)

    jCnt = 0
    while True:
        temp_sol[1:L-1, 1:M-1, 1:N-1] = ((hy2hz2*(U[0:L-2, 1:M-1, 1:N-1] + U[2:L, 1:M-1, 1:N-1]) +
                                        hz2hx2*(U[1:L-1, 0:M-2, 1:N-1] + U[1:L-1, 2:M, 1:N-1]) +
                                        hx2hy2*(U[1:L-1, 1:M-1, 0:N-2] + U[1:L-1, 1:M-1, 2:N]))*
                                nu*dt/(hx2hy2hz2*2.0) + rho[1:L-1, 1:M-1, 1:N-1])/ \
                            (1.0 + nu*dt*(hy2hz2 + hz2hx2 + hx2hy2)/(hx2hy2hz2))
        
            # SWAP ARRAYS AND IMPOSE BOUNDARY CONDITION
            #U, temp_sol = temp_sol, U
        U = temp_sol

        imposeUBCs(U)
        
        maxErr = np.amax(np.fabs(rho[1:L-1, 1:M-1, 1:N-1] - (U[1:L-1, 1:M-1, 1:N-1] - 0.5*nu*dt*(
                            (U[0:L-2, 1:M-1, 1:N-1] - 2.0*U[1:L-1, 1:M-1, 1:N-1] + U[2:L, 1:M-1, 1:N-1])/hx2 +
                            (U[1:L-1, 0:M-2, 1:N-1] - 2.0*U[1:L-1, 1:M-1, 1:N-1] + U[1:L-1, 2:M, 1:N-1])/hy2 +
                            (U[1:L-1, 1:M-1, 0:N-2] - 2.0*U[1:L-1, 1:M-1, 1:N-1] + U[1:L-1, 1:M-1, 2:N])/hz2))))
        
            #if maxErr < tolerance:
        if maxErr < VpTolerance:
            #print(jCnt)
            break
        
        jCnt += 1
        if jCnt > maxCount:
                print("ERROR: Jacobi not converging in U. Aborting")
                print("Maximum error: ", maxErr)
                quit()

    return U        


#Jacobi iterative solver for V
def vJacobi(rho):
    global hx2, hy2, hz2, hy2hz2, hz2hx2, hx2hy2, hx2hy2hz2, nu, dt, VpTolerance, maxCount  
    global V
    global L, M, N, Nx, Ny, Nz
    
    temp_sol = np.zeros_like(rho)
    
    jCnt = 0
    while True:
        temp_sol[1:L-1, 1:M-1, 1:N-1] = ((hy2hz2*(V[0:L-2, 1:M-1, 1:N-1] + V[2:L, 1:M-1, 1:N-1]) +
                                    hz2hx2*(V[1:L-1, 0:M-2, 1:N-1] + V[1:L-1, 2:M, 1:N-1]) +
                                    hx2hy2*(V[1:L-1, 1:M-1, 0:N-2] + V[1:L-1, 1:M-1, 2:N]))*
                                nu*dt/(hx2hy2hz2*2.0) + rho[1:L-1, 1:M-1, 1:N-1])/ \
                        (1.0 + nu*dt*(hy2hz2 + hz2hx2 + hx2hy2)/(hx2hy2hz2))
    
        # SWAP ARRAYS AND IMPOSE BOUNDARY CONDITION
        #V, temp_sol = temp_sol, V
        V = temp_sol

        imposeVBCs(V)


        maxErr = np.amax(np.fabs(rho[1:L-1, 1:M-1, 1:N-1] - (V[1:L-1, 1:M-1, 1:N-1] - 0.5*nu*dt*(
                        (V[0:L-2, 1:M-1, 1:N-1] - 2.0*V[1:L-1, 1:M-1, 1:N-1] + V[2:L, 1:M-1, 1:N-1])/hx2 +
                        (V[1:L-1, 0:M-2, 1:N-1] - 2.0*V[1:L-1, 1:M-1, 1:N-1] + V[1:L-1, 2:M, 1:N-1])/hy2 +
                        (V[1:L-1, 1:M-1, 0:N-2] - 2.0*V[1:L-1, 1:M-1, 1:N-1] + V[1:L-1, 1:M-1, 2:N])/hz2))))
    
        #if maxErr < tolerance:
        if maxErr < VpTolerance:
            #print(jCnt)
            break
    
        jCnt += 1
        if jCnt > maxCount:
            print("ERROR: Jacobi not converging in V. Aborting")
            print("Maximum error: ", maxErr)
            quit()
    
    return V



#Jacobi iterative solver for W
def wJacobi(rho):
    global hx2, hy2, hz2, hy2hz2, hz2hx2, hx2hy2, hx2hy2hz2, nu, dt, VpTolerance, maxCount  
    global W
    global L, M, N, Nx, Ny, Nz
    
    temp_sol = np.zeros_like(rho)
    
    jCnt = 0
    while True:
        temp_sol[1:L-1, 1:M-1, 1:N-1] = ((hy2hz2*(W[0:L-2, 1:M-1, 1:N-1] + W[2:L, 1:M-1, 1:N-1]) +
                                    hz2hx2*(W[1:L-1, 0:M-2, 1:N-1] + W[1:L-1, 2:M, 1:N-1]) +
                                    hx2hy2*(W[1:L-1, 1:M-1, 0:N-2] + W[1:L-1, 1:M-1, 2:N]))*
                                nu*dt/(hx2hy2hz2*2.0) + rho[1:L-1, 1:M-1, 1:N-1])/ \
                        (1.0 + nu*dt*(hy2hz2 + hz2hx2 + hx2hy2)/(hx2hy2hz2))
    
        # SWAP ARRAYS AND IMPOSE BOWNDARY CONDITION
        #W, temp_sol = temp_sol, W
    
        W = temp_sol
    
        imposeWBCs(W)


        maxErr = np.amax(np.fabs(rho[1:L-1, 1:M-1, 1:N-1] - (W[1:L-1, 1:M-1, 1:N-1] - 0.5*nu*dt*(
                        (W[0:L-2, 1:M-1, 1:N-1] - 2.0*W[1:L-1, 1:M-1, 1:N-1] + W[2:L, 1:M-1, 1:N-1])/hx2 +
                        (W[1:L-1, 0:M-2, 1:N-1] - 2.0*W[1:L-1, 1:M-1, 1:N-1] + W[1:L-1, 2:M, 1:N-1])/hy2 +
                        (W[1:L-1, 1:M-1, 0:N-2] - 2.0*W[1:L-1, 1:M-1, 1:N-1] + W[1:L-1, 1:M-1, 2:N])/hz2))))
    
        #if maxErr < tolerance:
        if maxErr < 1e-5:
            #print(jCnt)
            break
    
        jCnt += 1
        if jCnt > maxCount:
            print("ERROR: Jacobi not converging in W. Aborting")
            print("Maximum error: ", maxErr)
            quit()
    
    return W       


#Jacobi iterative solver for T
def TJacobi(rho):
    global hx2, hy2, hz2, hy2hz2, hz2hx2, hx2hy2, hx2hy2hz2, nu, dt, VpTolerance, maxCount  
    global T
    global L, M, N, Nx, Ny, Nz
    
    temp_sol = np.zeros_like(rho)
    
    jCnt = 0
    while True:
        temp_sol[1:L-1, 1:M-1, 1:N-1] = ((hy2hz2*(T[0:L-2, 1:M-1, 1:N-1] + T[2:L, 1:M-1, 1:N-1]) +
                                    hz2hx2*(T[1:L-1, 0:M-2, 1:N-1] + T[1:L-1, 2:M, 1:N-1]) +
                                    hx2hy2*(T[1:L-1, 1:M-1, 0:N-2] + T[1:L-1, 1:M-1, 2:N]))*
                                nu*dt/(hx2hy2hz2*2.0) + rho[1:L-1, 1:M-1, 1:N-1])/ \
                        (1.0 + nu*dt*(hy2hz2 + hz2hx2 + hx2hy2)/(hx2hy2hz2))
    
        # SWAP ARRAYS AND IMPOSE BOTNDARY CONDITION
        #T, temp_sol = temp_sol, T
        T = temp_sol

        imposeTBCs(T)

        maxErr = np.amax(np.fabs(rho[1:L-1, 1:M-1, 1:N-1] - (T[1:L-1, 1:M-1, 1:N-1] - 0.5*nu*dt*(
                        (T[0:L-2, 1:M-1, 1:N-1] - 2.0*T[1:L-1, 1:M-1, 1:N-1] + T[2:L, 1:M-1, 1:N-1])/hx2 +
                        (T[1:L-1, 0:M-2, 1:N-1] - 2.0*T[1:L-1, 1:M-1, 1:N-1] + T[1:L-1, 2:M, 1:N-1])/hy2 +
                        (T[1:L-1, 1:M-1, 0:N-2] - 2.0*T[1:L-1, 1:M-1, 1:N-1] + T[1:L-1, 1:M-1, 2:N])/hz2))))
    
        #if maxErr < tolerance:
        if maxErr < VpTolerance:
            #print(jCnt)
            break
    
        jCnt += 1
        if jCnt > maxCount:
            print("ERROR: Jacobi not converging in T. Aborting")
            print("Maximum error: ", maxErr)
            quit()
    
    return T       



def PoissonSolver(rho):
    global hx2, hy2, hz2, hy2hz2, hz2hx2, hx2hy2, hx2hy2hz2, nu, dt, PoissonTolerance, maxCount 
    global N, M, L    
    
    
    Pp = np.zeros([L, M, N])
    #Pp = np.random.rand(Nx, Ny, Nz)
    #Ppp = np.zeros([L, M, N])
    
    #print(np.amax(rho))
    
    jCnt = 0
    
    while True:

        '''
        
        for i in range(1,Nx-1):
            for j in range(1,Ny-1):
                for k in range(1,Nz-1):
                    Pp[i,j,k] = (1.0/(-2.0*(idx2 + idy2 + idz2))) * (rho[i, j, k] - 
                                       idx2*(Pp[i+1, j, k] + Pp[i-1, j, k]) -
                                       idy2*(Pp[i, j+1, k] + Pp[i, j-1, k]) -
                                       idz2*(Pp[i, j, k+1] + Pp[i, j, k-1]))

        Pp[1:L-1, 1:M-1, 1:N-1] = (1.0-gssor)*Ppp[1:L-1, 1:M-1, 1:N-1] + gssor * Pp[1:L-1, 1:M-1, 1:N-1]            

        '''
           
        
        Pp[1:L-1, 1:M-1, 1:N-1] = (1.0/(-2.0*(idx2 + idy2 + idz2))) * (rho[1:L-1, 1:M-1, 1:N-1] - 
                                       idx2*(Pp[0:L-2, 1:M-1, 1:N-1] + Pp[2:L, 1:M-1, 1:N-1]) -
                                       idy2*(Pp[1:L-1, 0:M-2, 1:N-1] + Pp[1:L-1, 2:M, 1:N-1]) -
                                       idz2*(Pp[1:L-1, 1:M-1, 0:N-2] + Pp[1:L-1, 1:M-1, 2:N]))   


        #Pp[1:L-1, 1:M-1, 1:N-1] = (1.0-gssor)*Ppp[1:L-1, 1:M-1, 1:N-1] + gssor*Pp[1:L-1, 1:M-1, 1:N-1]                                                                   
           
        #Ppp = Pp.copy()

        #imposePBCs(Pp)
    
        maxErr = np.amax(np.fabs(rho[1:L-1, 1:M-1, 1:N-1] -((
                        (Pp[0:L-2, 1:M-1, 1:N-1] - 2.0*Pp[1:L-1, 1:M-1, 1:N-1] + Pp[2:L, 1:M-1, 1:N-1])/hx2 +
                        (Pp[1:L-1, 0:M-2, 1:N-1] - 2.0*Pp[1:L-1, 1:M-1, 1:N-1] + Pp[1:L-1, 2:M, 1:N-1])/hy2 +
                        (Pp[1:L-1, 1:M-1, 0:N-2] - 2.0*Pp[1:L-1, 1:M-1, 1:N-1] + Pp[1:L-1, 1:M-1, 2:N])/hz2))))
    
    
        #if (jCnt % 100 == 0):
        #    print(maxErr)
    
        if maxErr < PoissonTolerance:
            print(jCnt)
            #print("Poisson solver converged")
            break
    
        jCnt += 1
        if jCnt > maxCount:
            print("ERROR: Poisson solver not converging. Aborting")
            quit()
    
    return Pp     


def imposeUBCs(U):
    U[0, :, :], U[-1, :, :] = 0.0, 0.0
    U[:, 0, :], U[:, -1, :] = 0.0, 0.0
    U[:, :, 0], U[:, :, -1] = 0.0, 0.0

def imposeVBCs(V):
    V[0, :, :], V[-1, :, :] = 0.0, 0.0  
    V[:, 0, :], V[:, -1, :] = 0.0, 0.0  
    V[:, :, 0], V[:, :, -1] = 0.0, 0.0

def imposeWBCs(W):
    W[0, :, :], W[-1, :, :] = 0.0, 0.0, 
    W[:, 0, :], W[:, -1, :] = 0.0, 0.0
    W[:, :, 0], W[:, :, -1] = 0.0, 0.0  

def imposeTBCs(T):
    T[0, :, :], T[-1, :, :] = T[1, :, :], T[-2, :, :]
    T[:, 0, :], T[:, -1, :] = T[:, 1, :], T[:, -2, :]
    T[:, :, 0], T[:, :, -1] = 1.0, 0.0

def imposePBCs(P):
    P[0, :, :], P[-1, :, :] = P[1, :, :], P[-2, :, :]
    P[:, 0, :], P[:, -1, :] = P[:, 1, :], P[:, -2, :]
    P[:, :, 0], P[:, :, -1] = P[:, :, 1], P[:, :, -2]


initFields()

TimeIntegrate()




