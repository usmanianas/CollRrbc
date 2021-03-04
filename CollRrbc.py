
import numpy as np
import pylab as plt
from matplotlib import cm
import h5py as hp
import time
from datetime import datetime
import random 



#### Grid Parameters ###########################
Lx, Ly, Lz = 1.0, 1.0, 1.0

Nx, Ny, Nz = 32, 32, 32

hx, hy, hz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)

x = np.linspace(0, 1, Nx, endpoint=True)        
y = np.linspace(0, 1, Ny, endpoint=True)
z = np.linspace(0, 1, Nz, endpoint=True)    

hx2, hy2, hz2 = hx*hx, hy*hy, hz*hz

hx2hy2, hy2hz2, hz2hx2 = hx2*hy2, hy2*hz2, hz2*hx2

hx2hy2hz2 = hx2*hy2*hz2

idx2, idy2, idz2 = 1.0/hx2, 1.0/hy2, 1.0/hz2
#############################################################




#### Flow Parameters #############
Ra = 1.0e5

Pr = 0.786

Ta = 0.0e7

print("#", "Ra=", Ra, "Pr=", Pr, "Ta=", Ta)

#Ro = np.sqrt(Ra/(Ta*Pr))

nu, kappa = np.sqrt(Pr/Ra), 1.0/np.sqrt(Ra*Pr)

#print(nu, kappa)

#########################################################




#########Simulation Parameters #########################
dt = 0.01

tMax = 1000

# Number of iterations after which output must be printed to standard I/O
opInt = 1

# File writing interval
fwInt = 2

# Tolerance value in Jacobi iterations
VpTolerance = 1.0e-5

# Tolerance value in Poisson iterations
PoissonTolerance = 1.0e-3

gssor = 1.6

maxCount = 1e4
#################################################


restart = 0    # 0-Fresh, 1-Restart

if restart == 1:
    filename = "Soln_500.00000.h5"
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

else:
    P = np.ones([Nx, Ny, Nz])

#T = random.uniform(0, 1) * np.ones([Nx, Ny, Nz])
    T = np.zeros([Nx, Ny, Nz])

    T[:, :, 0:Nz] = 1 - z[0:Nz]

    U = np.zeros([Nx, Ny, Nz])

    V = np.zeros([Nx, Ny, Nz])

    W = np.zeros([Nx, Ny, Nz])


Hx = np.zeros_like(U)
Hy = np.zeros_like(V)
Hz = np.zeros_like(W)
Ht = np.zeros_like(T)   
Pp = np.zeros_like(P)

Hx.fill(0.0)
Hy.fill(0.0)
Hz.fill(0.0)
Ht.fill(0.0)

time = 0
fwTime = 0.0
iCnt = 1


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

writeSoln(U, V, W, P, T, time)



def getDiv(U, V, W):

    divMat = np.zeros([Nx, Ny, Nz])

    divMat[1:Nx-1, 1:Ny-1, 1:Nz-1] = ((U[2:Nx, 1:Ny-1, 1:Nz-1] - U[0:Nx-2, 1:Ny-1, 1:Nz-1])*0.5/hx +
                                (V[1:Nx-1, 2:Ny, 1:Nz-1] - V[1:Nx-1, 0:Ny-2, 1:Nz-1])*0.5/hy +
                                (W[1:Nx-1, 1:Ny-1, 2:Nz] - W[1:Nx-1, 1:Ny-1, 0:Nz-2])*0.5/hz)
    
    #return np.unravel_index(divNyat.argmax(), divMat.shape), np.mean(divMat)
    return np.max(abs(divMat))







def computeNLinDiff_X(U, V, W):
    global Hx
    global Nz, Ny, Nx, Nx, Ny, Nz

    Hx[1:Nx-1, 1:Ny-1, 1:Nz-1] = (((U[2:Nx, 1:Ny-1, 1:Nz-1] - 2.0*U[1:Nx-1, 1:Ny-1, 1:Nz-1] + U[0:Nx-2, 1:Ny-1, 1:Nz-1])/hx2 + 
                                (U[1:Nx-1, 2:Ny, 1:Nz-1] - 2.0*U[1:Nx-1, 1:Ny-1, 1:Nz-1] + U[1:Nx-1, 0:Ny-2, 1:Nz-1])/hy2 + 
                                (U[1:Nx-1, 1:Ny-1, 2:Nz] - 2.0*U[1:Nx-1, 1:Ny-1, 1:Nz-1] + U[1:Nx-1, 1:Ny-1, 0:Nz-2])/hz2)*0.5*nu -
                              U[1:Nx-1, 1:Ny-1, 1:Nz-1]*(U[2:Nx, 1:Ny-1, 1:Nz-1] - U[0:Nx-2, 1:Ny-1, 1:Nz-1])/(2.0*hx) -
                              V[1:Nx-1, 1:Ny-1, 1:Nz-1]*(U[1:Nx-1, 2:Ny, 1:Nz-1] - U[1:Nx-1, 0:Ny-2, 1:Nz-1])/(2.0*hy) - 
                              W[1:Nx-1, 1:Ny-1, 1:Nz-1]*(U[1:Nx-1, 1:Ny-1, 2:Nz] - U[1:Nx-1, 1:Ny-1, 0:Nz-2])/(2.0*hz))

    return Hx

def computeNLinDiff_Y(U, V, W):
    global Hy
    global Nz, Ny, Nx, Nx, Ny, Nz

    Hy[1:Nx-1, 1:Ny-1, 1:Nz-1] = (((V[2:Nx, 1:Ny-1, 1:Nz-1] - 2.0*V[1:Nx-1, 1:Ny-1, 1:Nz-1] + V[0:Nx-2, 1:Ny-1, 1:Nz-1])/hx2 + 
                                (V[1:Nx-1, 2:Ny, 1:Nz-1] - 2.0*V[1:Nx-1, 1:Ny-1, 1:Nz-1] + V[1:Nx-1, 0:Ny-2, 1:Nz-1])/hy2 + 
                                (V[1:Nx-1, 1:Ny-1, 2:Nz] - 2.0*V[1:Nx-1, 1:Ny-1, 1:Nz-1] + V[1:Nx-1, 1:Ny-1, 0:Nz-2])/hz2)*0.5*nu -
                              U[1:Nx-1, 1:Ny-1, 1:Nz-1]*(V[2:Nx, 1:Ny-1, 1:Nz-1] - V[0:Nx-2, 1:Ny-1, 1:Nz-1])/(2.0*hx) -
                              V[1:Nx-1, 1:Ny-1, 1:Nz-1]*(V[1:Nx-1, 2:Ny, 1:Nz-1] - V[1:Nx-1, 0:Ny-2, 1:Nz-1])/(2.0*hy) - 
                              W[1:Nx-1, 1:Ny-1, 1:Nz-1]*(V[1:Nx-1, 1:Ny-1, 2:Nz] - V[1:Nx-1, 1:Ny-1, 0:Nz-2])/(2.0*hz))

    return Hy


def computeNLinDiff_Z(U, V, W):
    global Hz
    global Nz, Ny, Nx, Nx, Ny, Nz

    Hz[1:Nx-1, 1:Ny-1, 1:Nz-1] = (((W[2:Nx, 1:Ny-1, 1:Nz-1] - 2.0*W[1:Nx-1, 1:Ny-1, 1:Nz-1] + W[0:Nx-2, 1:Ny-1, 1:Nz-1])/hx2 + 
                                (W[1:Nx-1, 2:Ny, 1:Nz-1] - 2.0*W[1:Nx-1, 1:Ny-1, 1:Nz-1] + W[1:Nx-1, 0:Ny-2, 1:Nz-1])/hy2 + 
                                (W[1:Nx-1, 1:Ny-1, 2:Nz] - 2.0*W[1:Nx-1, 1:Ny-1, 1:Nz-1] + W[1:Nx-1, 1:Ny-1, 0:Nz-2])/hz2)*0.5*nu -
                              U[1:Nx-1, 1:Ny-1, 1:Nz-1]*(W[2:Nx, 1:Ny-1, 1:Nz-1] - W[0:Nx-2, 1:Ny-1, 1:Nz-1])/(2.0*hx) -
                              V[1:Nx-1, 1:Ny-1, 1:Nz-1]*(W[1:Nx-1, 2:Ny, 1:Nz-1] - W[1:Nx-1, 0:Ny-2, 1:Nz-1])/(2.0*hy) - 
                              W[1:Nx-1, 1:Ny-1, 1:Nz-1]*(W[1:Nx-1, 1:Ny-1, 2:Nz] - W[1:Nx-1, 1:Ny-1, 0:Nz-2])/(2.0*hz))


    return Hz


def computeNLinDiff_T(U, V, W, T):
    global Ht
    global Nz, Ny, Nx

    Ht[1:Nx-1, 1:Ny-1, 1:Nz-1] = (((T[2:Nx, 1:Ny-1, 1:Nz-1] - 2.0*T[1:Nx-1, 1:Ny-1, 1:Nz-1] + T[0:Nx-2, 1:Ny-1, 1:Nz-1])/hx2 + 
                                (T[1:Nx-1, 2:Ny, 1:Nz-1] - 2.0*T[1:Nx-1, 1:Ny-1, 1:Nz-1] + T[1:Nx-1, 0:Ny-2, 1:Nz-1])/hy2 + 
                                (T[1:Nx-1, 1:Ny-1, 2:Nz] - 2.0*T[1:Nx-1, 1:Ny-1, 1:Nz-1] + T[1:Nx-1, 1:Ny-1, 0:Nz-2])/hz2)*0.5*kappa -
                              U[1:Nx-1, 1:Ny-1, 1:Nz-1]*(T[2:Nx, 1:Ny-1, 1:Nz-1] - T[0:Nx-2, 1:Ny-1, 1:Nz-1])/(2.0*hx)-
                              V[1:Nx-1, 1:Ny-1, 1:Nz-1]*(T[1:Nx-1, 2:Ny, 1:Nz-1] - T[1:Nx-1, 0:Ny-2, 1:Nz-1])/(2.0*hy) - 
                              W[1:Nx-1, 1:Ny-1, 1:Nz-1]*(T[1:Nx-1, 1:Ny-1, 2:Nz] - T[1:Nx-1, 1:Ny-1, 0:Nz-2])/(2.0*hz))

    return Ht


#Jacobi iterative solver for U
def uJacobi(rho):
    global hx2, hy2, hz2, hy2hz2, hz2hx2, hx2hy2, hx2hy2hz2, nu, dt, VpTolerance, maxCount
    global U
    global Nx, Ny, Nz, Nx, Ny, Nz

    jCnt = 0
    while True:

        U[1:Nx-1, 1:Ny-1, 1:Nz-1] =(1.0/(1+nu*dt*(idx2 + idy2 + idz2))) * (rho[1:Nx-1, 1:Ny-1, 1:Nz-1] + 
                                       0.5*nu*dt*idx2*(U[0:Nx-2, 1:Ny-1, 1:Nz-1] + U[2:Nx, 1:Ny-1, 1:Nz-1]) +
                                       0.5*nu*dt*idy2*(U[1:Nx-1, 0:Ny-2, 1:Nz-1] + U[1:Nx-1, 2:Ny, 1:Nz-1]) +
                                       0.5*nu*dt*idz2*(U[1:Nx-1, 1:Ny-1, 0:Nz-2] + U[1:Nz-1, 1:Ny-1, 2:Nz]))          

        imposeUBCs(U)
        
        maxErr = np.amax(np.fabs(rho[1:Nx-1, 1:Ny-1, 1:Nz-1] - (U[1:Nx-1, 1:Ny-1, 1:Nz-1] - 0.5*nu*dt*(
                            (U[0:Nx-2, 1:Ny-1, 1:Nz-1] - 2.0*U[1:Nx-1, 1:Ny-1, 1:Nz-1] + U[2:Nx, 1:Ny-1, 1:Nz-1])/hx2 +
                            (U[1:Nx-1, 0:Ny-2, 1:Nz-1] - 2.0*U[1:Nx-1, 1:Ny-1, 1:Nz-1] + U[1:Nx-1, 2:Ny, 1:Nz-1])/hy2 +
                            (U[1:Nx-1, 1:Ny-1, 0:Nz-2] - 2.0*U[1:Nx-1, 1:Ny-1, 1:Nz-1] + U[1:Nx-1, 1:Ny-1, 2:Nz])/hz2))))
        
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
    global Nx, Ny, Nz, Nx, Ny, Nz
        
    jCnt = 0
    while True:

        V[1:Nx-1, 1:Ny-1, 1:Nz-1] =(1.0/(1+nu*dt*(idx2 + idy2 + idz2))) * (rho[1:Nx-1, 1:Ny-1, 1:Nz-1] + 
                                       0.5*nu*dt*idx2*(V[0:Nx-2, 1:Ny-1, 1:Nz-1] + V[2:Nx, 1:Ny-1, 1:Nz-1]) +
                                       0.5*nu*dt*idy2*(V[1:Nx-1, 0:Ny-2, 1:Nz-1] + V[1:Nx-1, 2:Ny, 1:Nz-1]) +
                                       0.5*nu*dt*idz2*(V[1:Nx-1, 1:Ny-1, 0:Nz-2] + V[1:Nz-1, 1:Ny-1, 2:Nz]))   
       
        imposeVBCs(V)


        maxErr = np.amax(np.fabs(rho[1:Nx-1, 1:Ny-1, 1:Nz-1] - (V[1:Nx-1, 1:Ny-1, 1:Nz-1] - 0.5*nu*dt*(
                        (V[0:Nx-2, 1:Ny-1, 1:Nz-1] - 2.0*V[1:Nx-1, 1:Ny-1, 1:Nz-1] + V[2:Nx, 1:Ny-1, 1:Nz-1])/hx2 +
                        (V[1:Nx-1, 0:Ny-2, 1:Nz-1] - 2.0*V[1:Nx-1, 1:Ny-1, 1:Nz-1] + V[1:Nx-1, 2:Ny, 1:Nz-1])/hy2 +
                        (V[1:Nx-1, 1:Ny-1, 0:Nz-2] - 2.0*V[1:Nx-1, 1:Ny-1, 1:Nz-1] + V[1:Nx-1, 1:Ny-1, 2:Nz])/hz2))))
    
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
    global Nx, Ny, Nz, Nx, Ny, Nz
    
    
    jCnt = 0
    while True:

        W[1:Nx-1, 1:Ny-1, 1:Nz-1] =(1.0/(1+nu*dt*(idx2 + idy2 + idz2))) * (rho[1:Nx-1, 1:Ny-1, 1:Nz-1] + 
                                       0.5*nu*dt*idx2*(W[0:Nx-2, 1:Ny-1, 1:Nz-1] + W[2:Nx, 1:Ny-1, 1:Nz-1]) +
                                       0.5*nu*dt*idy2*(W[1:Nx-1, 0:Ny-2, 1:Nz-1] + W[1:Nx-1, 2:Ny, 1:Nz-1]) +
                                       0.5*nu*dt*idz2*(W[1:Nx-1, 1:Ny-1, 0:Nz-2] + W[1:Nz-1, 1:Ny-1, 2:Nz]))           
    
        imposeWBCs(W)


        maxErr = np.amax(np.fabs(rho[1:Nx-1, 1:Ny-1, 1:Nz-1] - (W[1:Nx-1, 1:Ny-1, 1:Nz-1] - 0.5*nu*dt*(
                        (W[0:Nx-2, 1:Ny-1, 1:Nz-1] - 2.0*W[1:Nx-1, 1:Ny-1, 1:Nz-1] + W[2:Nx, 1:Ny-1, 1:Nz-1])/hx2 +
                        (W[1:Nx-1, 0:Ny-2, 1:Nz-1] - 2.0*W[1:Nx-1, 1:Ny-1, 1:Nz-1] + W[1:Nx-1, 2:Ny, 1:Nz-1])/hy2 +
                        (W[1:Nx-1, 1:Ny-1, 0:Nz-2] - 2.0*W[1:Nx-1, 1:Ny-1, 1:Nz-1] + W[1:Nx-1, 1:Ny-1, 2:Nz])/hz2))))
    
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
    global Nx, Ny, Nz, Nx, Ny, Nz
        
    jCnt = 0
    while True:

        T[1:Nx-1, 1:Ny-1, 1:Nz-1] =(1.0/(1+kappa*dt*(idx2 + idy2 + idz2))) * (rho[1:Nx-1, 1:Ny-1, 1:Nz-1] + 
                                       0.5*kappa*dt*idx2*(T[0:Nx-2, 1:Ny-1, 1:Nz-1] + T[2:Nx, 1:Ny-1, 1:Nz-1]) +
                                       0.5*kappa*dt*idy2*(T[1:Nx-1, 0:Ny-2, 1:Nz-1] + T[1:Nx-1, 2:Ny, 1:Nz-1]) +
                                       0.5*kappa*dt*idz2*(T[1:Nx-1, 1:Ny-1, 0:Nz-2] + T[1:Nz-1, 1:Ny-1, 2:Nz])) 

        imposeTBCs(T)

        maxErr = np.amax(np.fabs(rho[1:Nx-1, 1:Ny-1, 1:Nz-1] - (T[1:Nx-1, 1:Ny-1, 1:Nz-1] - 0.5*kappa*dt*(
                        (T[0:Nx-2, 1:Ny-1, 1:Nz-1] - 2.0*T[1:Nx-1, 1:Ny-1, 1:Nz-1] + T[2:Nx, 1:Ny-1, 1:Nz-1])/hx2 +
                        (T[1:Nx-1, 0:Ny-2, 1:Nz-1] - 2.0*T[1:Nx-1, 1:Ny-1, 1:Nz-1] + T[1:Nx-1, 2:Ny, 1:Nz-1])/hy2 +
                        (T[1:Nx-1, 1:Ny-1, 0:Nz-2] - 2.0*T[1:Nx-1, 1:Ny-1, 1:Nz-1] + T[1:Nx-1, 1:Ny-1, 2:Nz])/hz2))))
    
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
    global Nz, Ny, Nx    
    
    
    Pp = np.zeros([Nx, Ny, Nz])
    #Pp = np.random.rand(Nx, Ny, Nz)
    #Ppp = np.zeros([Nx, Ny, Nz])
        
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

        Pp[1:Nx-1, 1:Ny-1, 1:Nz-1] = (1.0-gssor)*Ppp[1:Nx-1, 1:Ny-1, 1:Nz-1] + gssor * Pp[1:Nx-1, 1:Ny-1, 1:Nz-1]            

        '''
           
        
        Pp[1:Nx-1, 1:Ny-1, 1:Nz-1] = (1.0/(-2.0*(idx2 + idy2 + idz2))) * (rho[1:Nx-1, 1:Ny-1, 1:Nz-1] - 
                                       idx2*(Pp[0:Nx-2, 1:Ny-1, 1:Nz-1] + Pp[2:Nx, 1:Ny-1, 1:Nz-1]) -
                                       idy2*(Pp[1:Nx-1, 0:Ny-2, 1:Nz-1] + Pp[1:Nx-1, 2:Ny, 1:Nz-1]) -
                                       idz2*(Pp[1:Nx-1, 1:Ny-1, 0:Nz-2] + Pp[1:Nx-1, 1:Ny-1, 2:Nz]))   


        #Pp[1:Nx-1, 1:Ny-1, 1:Nz-1] = (1.0-gssor)*Ppp[1:Nx-1, 1:Ny-1, 1:Nz-1] + gssor*Pp[1:Nx-1, 1:Ny-1, 1:Nz-1]                                                                   
           
        #Ppp = Pp.copy()

        #imposePBCs(Pp)
    
        maxErr = np.amax(np.fabs(rho[1:Nx-1, 1:Ny-1, 1:Nz-1] -((
                        (Pp[0:Nx-2, 1:Ny-1, 1:Nz-1] - 2.0*Pp[1:Nx-1, 1:Ny-1, 1:Nz-1] + Pp[2:Nx, 1:Ny-1, 1:Nz-1])/hx2 +
                        (Pp[1:Nx-1, 0:Ny-2, 1:Nz-1] - 2.0*Pp[1:Nx-1, 1:Ny-1, 1:Nz-1] + Pp[1:Nx-1, 2:Ny, 1:Nz-1])/hy2 +
                        (Pp[1:Nx-1, 1:Ny-1, 0:Nz-2] - 2.0*Pp[1:Nx-1, 1:Ny-1, 1:Nz-1] + Pp[1:Nx-1, 1:Ny-1, 2:Nz])/hz2))))
    
    
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



while True:

    t1 = datetime.now()

    if iCnt % opInt == 0:

        Re = np.mean(np.sqrt(U[1:Nx-1, 1:Ny-1, 1:Nz-1]**2.0 + V[1:Nx-1, 1:Ny-1, 1:Nz-1]**2.0 + W[1:Nx-1, 1:Ny-1, 1:Nz-1]**2.0))/nu
        Nu = 1.0 + np.mean(W[1:Nx-1, 1:Ny-1, 1:Nz-1]*T[1:Nx-1, 1:Ny-1, 1:Nz-1])/kappa
        maxDiv = getDiv(U, V, W)

        print("%f    %f    %f    %f" %(time, Re, Nu, maxDiv))           


    Hx = computeNLinDiff_X(U, V, W)
    Hy = computeNLinDiff_Y(U, V, W)
    Hz = computeNLinDiff_Z(U, V, W)
    Ht = computeNLinDiff_T(U, V, W, T)  

    Hx[1:Nx-1, 1:Ny-1, 1:Nz-1] = U[1:Nx-1, 1:Ny-1, 1:Nz-1] + dt*(Hx[1:Nx-1, 1:Ny-1, 1:Nz-1] - np.sqrt((Ta*Pr)/Ra)*(-V[1:Nx-1, 1:Ny-1, 1:Nz-1]) - (P[2:Nx, 1:Ny-1, 1:Nz-1] - P[0:Nx-2, 1:Ny-1, 1:Nz-1])/(2.0*hx))
    uJacobi(Hx)

    Hy[1:Nx-1, 1:Ny-1, 1:Nz-1] = V[1:Nx-1, 1:Ny-1, 1:Nz-1] + dt*(Hy[1:Nx-1, 1:Ny-1, 1:Nz-1] - np.sqrt((Ta*Pr)/Ra)*(U[1:Nx-1, 1:Ny-1, 1:Nz-1]) - (P[1:Nx-1, 2:Ny, 1:Nz-1] - P[1:Nx-1, 0:Ny-2, 1:Nz-1])/(2.0*hy))
    vJacobi(Hy)

    Hz[1:Nx-1, 1:Ny-1, 1:Nz-1] = W[1:Nx-1, 1:Ny-1, 1:Nz-1] + dt*(Hz[1:Nx-1, 1:Ny-1, 1:Nz-1] - ((P[1:Nx-1, 1:Ny-1, 2:Nz] - P[1:Nx-1, 1:Ny-1, 0:Nz-2])/(2.0*hz)) + T[1:Nx-1, 1:Ny-1, 1:Nz-1])
    wJacobi(Hz)

    Ht[1:Nx-1, 1:Ny-1, 1:Nz-1] = T[1:Nx-1, 1:Ny-1, 1:Nz-1] + dt*Ht[1:Nx-1, 1:Ny-1, 1:Nz-1]
    TJacobi(Ht)   

    rhs = np.zeros([Nx, Ny, Nz])
    rhs[1:Nx-1, 1:Ny-1, 1:Nz-1] = ((U[2:Nx, 1:Ny-1, 1:Nz-1] - U[0:Nx-2, 1:Ny-1, 1:Nz-1])/(2.0*hx) +
                                (V[1:Nx-1, 2:Ny, 1:Nz-1] - V[1:Nx-1, 0:Ny-2, 1:Nz-1])/(2.0*hy) +
                                (W[1:Nx-1, 1:Ny-1, 2:Nz] - W[1:Nx-1, 1:Ny-1, 0:Nz-2])/(2.0*hz))/dt

    tp1 = datetime.now()
    Pp = PoissonSolver(rhs)
    tp2 = datetime.now()
    #print(tp2-tp1)    

    P = P + Pp

    U[1:Nx-1, 1:Ny-1, 1:Nz-1] = U[1:Nx-1, 1:Ny-1, 1:Nz-1] - dt*(Pp[2:Nx, 1:Ny-1, 1:Nz-1] - Pp[0:Nx-2, 1:Ny-1, 1:Nz-1])/(2.0*hx)
    V[1:Nx-1, 1:Ny-1, 1:Nz-1] = V[1:Nx-1, 1:Ny-1, 1:Nz-1] - dt*(Pp[1:Nx-1, 2:Ny, 1:Nz-1] - Pp[1:Nx-1, 0:Ny-2, 1:Nz-1])/(2.0*hy)
    W[1:Nx-1, 1:Ny-1, 1:Nz-1] = W[1:Nx-1, 1:Ny-1, 1:Nz-1] - dt*(Pp[1:Nx-1, 1:Ny-1, 2:Nz] - Pp[1:Nx-1, 1:Ny-1, 0:Nz-2])/(2.0*hz)

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

    t2 = datetime.now()

    #print(t2-t1)










