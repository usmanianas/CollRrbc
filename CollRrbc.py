
import numpy as np
import pylab as plt
from matplotlib import cm
import h5py as hp
import time
from datetime import datetime
import random 
import scipy.integrate as integrate



#### Grid Parameters ###########################
Lx, Ly, Lz = 1.0, 1.0, 1.0

Nx = 33
Ny, Nz = Nx, Nx

hx, hy, hz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)

x = np.linspace(0, 1, Nx, endpoint=True)        
y = np.linspace(0, 1, Ny, endpoint=True)
z = np.linspace(0, 1, Nz, endpoint=True)    

hx2, hy2, hz2 = hx*hx, hy*hy, hz*hz

idx2, idy2, idz2 = 1.0/hx2, 1.0/hy2, 1.0/hz2

print('# Grid', Nx, Ny, Nz)
#############################################################



#### Flow Parameters #############
Ra = 1.0e4

Pr = 1

Ta = 0e5

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

# Solution File writing interval
fwInt = 50

# Restart File writing interval
rwInt = 10

# Tolerance value in Jacobi iterations
VpTolerance = 1.0e-5

# Tolerance value in Poisson iterations
PoissonTolerance = 1.0e-5

gssor = 1.0

maxCount = 1e4

print('# Tolerance', VpTolerance, PoissonTolerance)
#################################################


restart = 0   # 0-Fresh, 1-Restart

if restart == 1:
    filename = "Restart.h5"
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

    f = hp.File(filename, 'r')
    time = f.get('Time')
    time = np.array(time)
    f.close()

else:
    time = 0

    P = np.zeros([Nx, Ny, Nz])

    T = np.zeros([Nx, Ny, Nz])

    T[:, :, 0:Nz] = 1 - z[0:Nz]

    U = np.zeros([Nx, Ny, Nz]) #np.random.rand(Nx, Ny, Nz) #

    V = np.zeros([Nx, Ny, Nz]) #np.random.rand(Nx, Ny, Nz) #

    W = np.zeros([Nx, Ny, Nz]) #np.random.rand(Nx, Ny, Nz) #


Pp = np.zeros([Nx, Ny, Nz])
divMat = np.zeros([Nx, Ny, Nz])


Hx = np.zeros_like(U)
Hy = np.zeros_like(V)
Hz = np.zeros_like(W)
Ht = np.zeros_like(T)   
Pp = np.zeros_like(P)

Hx.fill(0.0)
Hy.fill(0.0)
Hz.fill(0.0)
Ht.fill(0.0)

fwTime = 0.0
rwTime = 0.0

if restart == 1:
    fwTime = time
    rwTime = time

iCnt = 1

def writeSoln(U, V, W, P, T, time):

    fName = "Soln_" + "{0:09.5f}.h5".format(time)
    print("#Writing solution file: ", fName)        
    f = hp.File(fName, "w")

    dset = f.create_dataset("Vx", data = U[1:Nx-1, 1:Ny-1, 1:Nz-1])
    dset = f.create_dataset("Vy", data = V[1:Nx-1, 1:Ny-1, 1:Nz-1])
    dset = f.create_dataset("Vz", data = W[1:Nx-1, 1:Ny-1, 1:Nz-1])
    dset = f.create_dataset("T", data = T[1:Nx-1, 1:Ny-1, 1:Nz-1])
    dset = f.create_dataset("P", data = P[1:Nx-1, 1:Ny-1, 1:Nz-1])
    dset = f.create_dataset("Time", data = time)
    f.close()


def writeRestart(U, V, W, P, T, time):

    fName = "Restart.h5"
    print("#Writing Restart file: ", fName)        
    f = hp.File(fName, "w")

    dset = f.create_dataset("U", data = U)
    dset = f.create_dataset("V", data = V)
    dset = f.create_dataset("W", data = W)
    dset = f.create_dataset("T", data = T)
    dset = f.create_dataset("P", data = P)
    dset = f.create_dataset("Time", data = time)
    f.close()


def getDiv(U, V, W):

    divMat[1:Nx-1, 1:Ny-1, 1:Nz-1] = ((U[2:Nx, 1:Ny-1, 1:Nz-1] - U[0:Nx-2, 1:Ny-1, 1:Nz-1])*0.5/hx +
                                (V[1:Nx-1, 2:Ny, 1:Nz-1] - V[1:Nx-1, 0:Ny-2, 1:Nz-1])*0.5/hy +
                                (W[1:Nx-1, 1:Ny-1, 2:Nz] - W[1:Nx-1, 1:Ny-1, 0:Nz-2])*0.5/hz)
    
    #return np.unravel_index(divNyat.argmax(), divMat.shape), np.mean(divMat)
    return np.max(abs(divMat))


def computeNLinDiff_X(U, V, W):

    Hx[1:Nx-1, 1:Ny-1, 1:Nz-1] = (((U[2:Nx, 1:Ny-1, 1:Nz-1] - 2.0*U[1:Nx-1, 1:Ny-1, 1:Nz-1] + U[0:Nx-2, 1:Ny-1, 1:Nz-1])/hx2 + 
                                (U[1:Nx-1, 2:Ny, 1:Nz-1] - 2.0*U[1:Nx-1, 1:Ny-1, 1:Nz-1] + U[1:Nx-1, 0:Ny-2, 1:Nz-1])/hy2 + 
                                (U[1:Nx-1, 1:Ny-1, 2:Nz] - 2.0*U[1:Nx-1, 1:Ny-1, 1:Nz-1] + U[1:Nx-1, 1:Ny-1, 0:Nz-2])/hz2)*0.5*nu -
                              U[1:Nx-1, 1:Ny-1, 1:Nz-1]*(U[2:Nx, 1:Ny-1, 1:Nz-1] - U[0:Nx-2, 1:Ny-1, 1:Nz-1])/(2.0*hx) -
                              V[1:Nx-1, 1:Ny-1, 1:Nz-1]*(U[1:Nx-1, 2:Ny, 1:Nz-1] - U[1:Nx-1, 0:Ny-2, 1:Nz-1])/(2.0*hy) - 
                              W[1:Nx-1, 1:Ny-1, 1:Nz-1]*(U[1:Nx-1, 1:Ny-1, 2:Nz] - U[1:Nx-1, 1:Ny-1, 0:Nz-2])/(2.0*hz))

    return Hx

def computeNLinDiff_Y(U, V, W):

    Hy[1:Nx-1, 1:Ny-1, 1:Nz-1] = (((V[2:Nx, 1:Ny-1, 1:Nz-1] - 2.0*V[1:Nx-1, 1:Ny-1, 1:Nz-1] + V[0:Nx-2, 1:Ny-1, 1:Nz-1])/hx2 + 
                                (V[1:Nx-1, 2:Ny, 1:Nz-1] - 2.0*V[1:Nx-1, 1:Ny-1, 1:Nz-1] + V[1:Nx-1, 0:Ny-2, 1:Nz-1])/hy2 + 
                                (V[1:Nx-1, 1:Ny-1, 2:Nz] - 2.0*V[1:Nx-1, 1:Ny-1, 1:Nz-1] + V[1:Nx-1, 1:Ny-1, 0:Nz-2])/hz2)*0.5*nu -
                              U[1:Nx-1, 1:Ny-1, 1:Nz-1]*(V[2:Nx, 1:Ny-1, 1:Nz-1] - V[0:Nx-2, 1:Ny-1, 1:Nz-1])/(2.0*hx) -
                              V[1:Nx-1, 1:Ny-1, 1:Nz-1]*(V[1:Nx-1, 2:Ny, 1:Nz-1] - V[1:Nx-1, 0:Ny-2, 1:Nz-1])/(2.0*hy) - 
                              W[1:Nx-1, 1:Ny-1, 1:Nz-1]*(V[1:Nx-1, 1:Ny-1, 2:Nz] - V[1:Nx-1, 1:Ny-1, 0:Nz-2])/(2.0*hz))

    return Hy


def computeNLinDiff_Z(U, V, W):

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


def uJacobi(rho):

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

        jCnt += 1        
        if maxErr < VpTolerance:
            #print(jCnt)
            break
        
        if jCnt > maxCount:
                print("ERROR: Jacobi not converging in U. Aborting")
                print("Maximum error: ", maxErr)
                quit()

    return U        


def vJacobi(rho):
        
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
        
        jCnt += 1    
        if maxErr < VpTolerance:
            #print(jCnt)
            break
    
        if jCnt > maxCount:
            print("ERROR: Jacobi not converging in V. Aborting")
            print("Maximum error: ", maxErr)
            quit()
    
    return V


def wJacobi(rho):    
    
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
        
        jCnt += 1
        if maxErr < VpTolerance:
            #print(jCnt)
            break
    
        if jCnt > maxCount:
            print("ERROR: Jacobi not converging in W. Aborting")
            print("Maximum error: ", maxErr)
            quit()
    
    return W       


def TJacobi(rho):
        
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

        jCnt += 1    
        if maxErr < VpTolerance:
            #print(jCnt)
            break
    
        if jCnt > maxCount:
            print("ERROR: Jacobi not converging in T. Aborting")
            print("Maximum error: ", maxErr)
            quit()
    
    return T       



def PoissonSolver(rho):   
    #Ppp = np.zeros([Nx, Ny, Nz])
    Pp = np.zeros([Nx, Ny, Nz])
        
    jCnt = 0   
    while True:
        
        '''
        #Ppp = Pp.copy()
        for i in range(1,Nx-1):
            for j in range(1,Ny-1):
                for k in range(1,Nz-1):
                    Pp[i,j,k] = (1.0-gssor)*Ppp[i,j,k] + (gssor/(-2.0*(idx2 + idy2 + idz2))) * (rho[i, j, k] - 
                                       idx2*(Pp[i+1, j, k] + Pp[i-1, j, k]) -
                                       idy2*(Pp[i, j+1, k] + Pp[i, j-1, k]) -
                                       idz2*(Pp[i, j, k+1] + Pp[i, j, k-1]))                    
        '''       


        #print(np.amax(rho), maxErr)

        Pp[1:Nx-1, 1:Ny-1, 1:Nz-1] = (1.0/(-2.0*(idx2 + idy2 + idz2))) * (rho[1:Nx-1, 1:Ny-1, 1:Nz-1] - 
                                       idx2*(Pp[0:Nx-2, 1:Ny-1, 1:Nz-1] + Pp[2:Nx, 1:Ny-1, 1:Nz-1]) -
                                       idy2*(Pp[1:Nx-1, 0:Ny-2, 1:Nz-1] + Pp[1:Nx-1, 2:Ny, 1:Nz-1]) -
                                       idz2*(Pp[1:Nx-1, 1:Ny-1, 0:Nz-2] + Pp[1:Nx-1, 1:Ny-1, 2:Nz]))   
                  
        imposePpBCs(Pp)
   
        maxErr = np.amax(np.fabs(rho[1:Nx-1, 1:Ny-1, 1:Nz-1] -((
                        (Pp[0:Nx-2, 1:Ny-1, 1:Nz-1] - 2.0*Pp[1:Nx-1, 1:Ny-1, 1:Nz-1] + Pp[2:Nx, 1:Ny-1, 1:Nz-1])/hx2 +
                        (Pp[1:Nx-1, 0:Ny-2, 1:Nz-1] - 2.0*Pp[1:Nx-1, 1:Ny-1, 1:Nz-1] + Pp[1:Nx-1, 2:Ny, 1:Nz-1])/hy2 +
                        (Pp[1:Nx-1, 1:Ny-1, 0:Nz-2] - 2.0*Pp[1:Nx-1, 1:Ny-1, 1:Nz-1] + Pp[1:Nx-1, 1:Ny-1, 2:Nz])/hz2))))

    
    
        #if (jCnt % 100 == 0):
        #    print(jCnt, maxErr)

        jCnt += 1
    
        if maxErr < PoissonTolerance:
            print(jCnt)
            break
    
        if jCnt > maxCount:
            print("ERROR: Poisson solver not converging. Aborting")
            quit()
    
    return Pp     


'''
def imposeUBCs(U):
    U[0, :, :], U[-1, :, :] = 0, 0
    U[:, 0, :], U[:, -1, :] = 0, 0
    U[:, :, 0], U[:, :, -1] = 0, 0
def imposeVBCs(V):
    V[0, :, :], V[-1, :, :] = 0, 0
    V[:, 0, :], V[:, -1, :] = 0, 0
    V[:, :, 0], V[:, :, -1] = 0, 0

def imposeWBCs(W):
    W[0, :, :], W[-1, :, :] = 0, 0
    W[:, 0, :], W[:, -1, :] = 0, 0
    W[:, :, 0], W[:, :, -1] = 0, 0
def imposeTBCs(T):
    T[0, :, :], T[-1, :, :] = T[1, :, :], T[-2, :, :]
    T[:, 0, :], T[:, -1, :] = T[:, 1, :], T[:, -2, :]
    T[:, :, 0], T[:, :, -1] = 1, 0

def imposePBCs(P):
    P[0, :, :], P[-1, :, :] = P[1, :, :], P[-2, :, :]
    P[:, 0, :], P[:, -1, :] = P[:, 1, :], P[:, -2, :]
    P[:, :, 0], P[:, :, -1] = P[:, :, 1], P[:, :, -2]

def imposePpBCs(Pp):
    Pp[0, :, :], Pp[-1, :, :] = 0, 0 #Pp[1, :, :], Pp[-2, :, :]
    Pp[:, 0, :], Pp[:, -1, :] = 0, 0 #Pp[:, 1, :], Pp[:, -2, :]
    Pp[:, :, 0], Pp[:, :, -1] = 0, 0 #Pp[:, :, 1], Pp[:, :, -2]    
'''


def imposeUBCs(U):
    U[0, :, :], U[-1, :, :] = -U[1, :, :], -U[-2, :, :]
    U[:, 0, :], U[:, -1, :] = -U[:, 1, :], -U[:, -2, :]
    U[:, :, 0], U[:, :, -1] = -U[:, :, 1], -U[:, :, -2]

def imposeVBCs(V):
    V[0, :, :], V[-1, :, :] = -V[1, :, :], -V[-2, :, :]
    V[:, 0, :], V[:, -1, :] = -V[:, 1, :], -V[:, -2, :]
    V[:, :, 0], V[:, :, -1] = -V[:, :, 1], -V[:, :, -2]

def imposeWBCs(W):
    W[0, :, :], W[-1, :, :] = -W[1, :, :], -W[-2, :, :]
    W[:, 0, :], W[:, -1, :] = -W[:, 1, :], -W[:, -2, :]
    W[:, :, 0], W[:, :, -1] = -W[:, :, 1], -W[:, :, -2]

def imposeTBCs(T):
    T[0, :, :], T[-1, :, :] = T[1, :, :], T[-2, :, :]
    T[:, 0, :], T[:, -1, :] = T[:, 1, :], T[:, -2, :]
    T[:, :, 0], T[:, :, -1] = 2.0 - T[:, :, 1], -T[:, :, -2]

def imposePBCs(P):
    P[0, :, :], P[-1, :, :] = P[1, :, :], P[-2, :, :]
    P[:, 0, :], P[:, -1, :] = P[:, 1, :], P[:, -2, :]
    P[:, :, 0], P[:, :, -1] = P[:, :, 1], P[:, :, -2]

def imposePpBCs(Pp):
    Pp[0, :, :], Pp[-1, :, :] = Pp[1, :, :], Pp[-2, :, :]
    Pp[:, 0, :], Pp[:, -1, :] = Pp[:, 1, :], Pp[:, -2, :]
    Pp[:, :, 0], Pp[:, :, -1] = Pp[:, :, 1], Pp[:, :, -2]        



while True:

    t1 = datetime.now()

    imposeUBCs(U)                               
    imposeVBCs(V)                               
    imposeWBCs(W)                               
    imposePBCs(P)                               
    imposeTBCs(T)       


    if iCnt % opInt == 0:
        uSqr = U[1:Nx-1, 1:Ny-1, 1:Nz-1]**2.0 + V[1:Nx-1, 1:Ny-1, 1:Nz-1]**2.0 + W[1:Nx-1, 1:Ny-1, 1:Nz-1]**2.0
        uInt = integrate.simps(integrate.simps(integrate.simps(uSqr, x[1:Nx-1]), y[1:Ny-1]), z[1:Nz-1])
        Re = np.sqrt(uInt)/nu

        wT = W[1:Nx-1, 1:Ny-1, 1:Nz-1]*T[1:Nx-1, 1:Ny-1, 1:Nz-1]
        wTInt = integrate.simps(integrate.simps(integrate.simps(wT, x[1:Nx-1]), y[1:Ny-1]), z[1:Nz-1])
        Nu = 1.0 + wTInt/kappa

        maxDiv = getDiv(U, V, W)

        f = open('TimeSeries.dat', 'a')
        if iCnt == 1:
            f.write("# %f \t %f \t %i \t %i \t %i \n" %(Ra, Pr, Nx, Ny, Nz)) 
            f.write('# time, Re, Nu, Divergence \n')

        f.write("%f \t %f \t %f \t %f \n" %(time, Re, Nu, maxDiv))
        #f.close()

        print("%f \t %.8f \t %.8f \t %.8f" %(time, Re, Nu, maxDiv))           



    if  abs(rwTime - time) < 0.5*dt:
        writeRestart(U, V, W, P, T, time)
        rwTime = rwTime + rwInt

    if abs(fwTime - time) < 0.5*dt:
        writeSoln(U, V, W, P, T, time)
        fwTime = fwTime + fwInt  

    if abs(time - tMax)<1e-5:
        Z, Y = np.meshgrid(y,z)
        plt.contourf(Y, Z, T[int(Nx/2), :, :], 500, cmap=cm.coolwarm)
        clb = plt.colorbar()
        plt.quiver(Y[0:Nx,0:Ny], Z[0:Nx,0:Ny], V[int(Nx/2),:, :], W[int(Nx/2), :, :])
        plt.axis('scaled')
        clb.ax.set_title(r'$T$', fontsize = 20)
        plt.show()


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

    #imposePpBCs(Pp)

    U[1:Nx-1, 1:Ny-1, 1:Nz-1] = U[1:Nx-1, 1:Ny-1, 1:Nz-1] - dt*(Pp[2:Nx, 1:Ny-1, 1:Nz-1] - Pp[0:Nx-2, 1:Ny-1, 1:Nz-1])/(2.0*hx)
    V[1:Nx-1, 1:Ny-1, 1:Nz-1] = V[1:Nx-1, 1:Ny-1, 1:Nz-1] - dt*(Pp[1:Nx-1, 2:Ny, 1:Nz-1] - Pp[1:Nx-1, 0:Ny-2, 1:Nz-1])/(2.0*hy)
    W[1:Nx-1, 1:Ny-1, 1:Nz-1] = W[1:Nx-1, 1:Ny-1, 1:Nz-1] - dt*(Pp[1:Nx-1, 1:Ny-1, 2:Nz] - Pp[1:Nx-1, 1:Ny-1, 0:Nz-2])/(2.0*hz)

 
    if abs(time - tMax) < 0.5*dt:
        print("Simulation completed!")
        break   

    time = time + dt

    iCnt = iCnt + 1

    t2 = datetime.now()

    #print(t2-t1)










