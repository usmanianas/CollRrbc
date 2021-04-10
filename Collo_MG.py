
import numpy as np
import pylab as plt
from matplotlib import cm
import h5py as hp
import time
from datetime import datetime
import random 
import scipy.integrate as integrate


#### Flow Parameters #############
Ra = 1e4

Pr = 0.71   #0.786 #0.71

Ta = 0e5

print()
print("#", "Ra=%.1e" %Ra, "Pr=%.3f" %Pr, "Ta=%.1e" %Ta)

#Ro = np.sqrt(Ra/(Ta*Pr))

nu, kappa = np.sqrt(Pr/Ra), 1.0/np.sqrt(Ra*Pr)

#print(nu, kappa)

#########################################################


#### Grid Parameters ###########################
Lx, Ly, Lz = 1.0, 1.0, 1.0

# Size index: 0 1 2 3  4  5  6  7   8   9   10   11   12   13    14
# Grid sizes: 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384

sInd = np.array([5, 5, 5])

#######################################


#########Simulation Parameters #########################
dt = 0.01

tMax = 1000

# Number of iterations after which output must be printed to standard I/O
opInt = 1

# Solution File writing interval
fwInt = 100

# Restart File writing interval
rwInt = 50

# Tolerance value in Jacobi iterations
VpTolerance = 1.0e-5

# Tolerance value in Poisson iterations
PoissonTolerance = 1.0e-5

gssor = 1.0

maxCount = 1e4

print('# Tolerance', VpTolerance, PoissonTolerance)
#################################################


restart = 0   # 0-Fresh, 1-Restart


###############Multigrid Parameters########################

# Depth of each V-cycle in multigrid
VDepth = 2 #min(sInd) - 1

# Number of iterations during pre-smoothing
preSm = 5

# Number of iterations during post-smoothing
pstSm = 20




# N should be of the form 2^n
# Then there will be 2^n + 2 points, including two ghost points
sLst = [2**x for x in range(12)]

Nx, Ny, Nz = sLst[sInd[0]] + 2, sLst[sInd[1]] + 2, sLst[sInd[2]] + 2

hx, hy, hz = Lx/(Nx-2), Ly/(Ny-2), Lz/(Nz-2)

x = np.linspace(-hx/2, Lx + hx/2, Nx, endpoint=True)        
y = np.linspace(-hy/2, Ly + hy/2, Ny, endpoint=True)
z = np.linspace(-hz/2, Lz + hz/2, Nz, endpoint=True)

hx2, hy2, hz2 = hx*hx, hy*hy, hz*hz

idx2, idy2, idz2 = 1.0/hx2, 1.0/hy2, 1.0/hz2

print('# Grid', Nx, Ny, Nz)
print('# Aspect ratio =',Ly/Lz)
print()
#############################################################


# Get array of grid sizes are tuples corresponding to each level of V-Cycle
N = [(sLst[x[0]], sLst[x[1]], sLst[x[2]]) for x in [sInd - y for y in range(VDepth + 1)]]

# Define array of grid spacings along X
h0 = Lx/(N[0][0])
mghx = [h0*(2**x) for x in range(VDepth+1)]

# Define array of grid spacings along Y
h0 = Ly/(N[0][1])
mghy = [h0*(2**x) for x in range(VDepth+1)]

# Define array of grid spacings along Z
h0 = Lz/(N[0][2])
mghz = [h0*(2**x) for x in range(VDepth+1)]

# Square of hx, used in finite difference formulae
mghx2 = [x*x for x in mghx]

# Square of hy, used in finite difference formulae
mghy2 = [x*x for x in mghy]

# Square of hz, used in finite difference formulae
mghz2 = [x*x for x in mghz]

# Cross product of hy and hz, used in finite difference formulae
hyhz = [mghy2[i]*mghz2[i] for i in range(VDepth + 1)]

# Cross product of hx and hz, used in finite difference formulae
hzhx = [mghx2[i]*mghz2[i] for i in range(VDepth + 1)]

# Cross product of hx and hy, used in finite difference formulae
hxhy = [mghx2[i]*mghy2[i] for i in range(VDepth + 1)]

# Cross product of hx, hy and hz used in finite difference formulae
hxhyhz = [mghx2[i]*mghy2[i]*mghz2[i] for i in range(VDepth + 1)]

# Factor in denominator of Gauss-Seidel iterations
gsFactor = [1.0/(2.0*(hyhz[i] + hzhx[i] + hxhy[i])) for i in range(VDepth + 1)]

# Maximum number of iterations while solving at coarsest level
maxCount = 10*N[-1][0]*N[-1][1]*N[-1][2]

# Integer specifying the level of V-cycle at any point while solving
vLev = 0


nList = np.array(N)

pData = [np.zeros(tuple(x)) for x in nList + 2]

rData = [np.zeros_like(x) for x in pData]
sData = [np.zeros_like(x) for x in pData]
iTemp = [np.zeros_like(x) for x in pData]

mgRHS = np.ones_like(pData[0])

# Smoothens the solution sCount times using Gauss-Seidel smoother
def smooth(sCount):
	global N
	global vLev
	global gsFactor
	global rData, pData
	global hyhz, hzhx, hxhy, hxhyhz

	#print(vLev)

	n = N[vLev]
	for iCnt in range(sCount):
		
		imposePpBCs(pData[vLev])
		
		
		pData[vLev][1:n[0]+1, 1:n[1]+1, 1:n[2]+1] = (hyhz[vLev]*(pData[vLev][2:n[0]+2, 1:n[1]+1, 1:n[2]+1] + pData[vLev][0:n[0], 1:n[1]+1, 1:n[2]+1]) +
													 hzhx[vLev]*(pData[vLev][1:n[0]+1, 2:n[1]+2, 1:n[2]+1] + pData[vLev][1:n[0]+1, 0:n[1], 1:n[2]+1]) +
													 hxhy[vLev]*(pData[vLev][1:n[0]+1, 1:n[1]+1, 2:n[2]+2] + pData[vLev][1:n[0]+1, 1:n[1]+1, 0:n[2]]) -
													 hxhyhz[vLev]*rData[vLev][0:n[0], 0:n[1], 0:n[2]]) * gsFactor[vLev]

		'''
		# Gauss-Seidel smoothing
		for i in range(1, n[0]+1):
			for j in range(1, n[1]+1):
				for k in range(1, n[2]+1):
					pData[vLev][i, j, k] = (hyhz[vLev]*(pData[vLev][i+1, j, k] + pData[vLev][i-1, j, k]) +
											hzhx[vLev]*(pData[vLev][i, j+1, k] + pData[vLev][i, j-1, k]) +
											hxhy[vLev]*(pData[vLev][i, j, k+1] + pData[vLev][i, j, k-1]) -
										  hxhyhz[vLev]*rData[vLev][i-1, j-1, k-1]) * gsFactor[vLev]
		'''

	imposePpBCs(pData[vLev])


# Compute the residual and store it into iTemp array
def calcResidual():
	global vLev
	global iTemp, rData, pData

	iTemp[vLev].fill(0.0)
	iTemp[vLev] = rData[vLev] - laplace(pData[vLev])


# Reduces the size of the array to a lower level, 2^(n - 1) + 1
def restrict():
	global N
	global vLev
	global iTemp, rData

	pLev = vLev
	vLev += 1

	n = N[vLev]
	rData[vLev] = (iTemp[pLev][::2, ::2, ::2] + iTemp[pLev][1::2, 1::2, 1::2] +
				   iTemp[pLev][::2, ::2, 1::2] + iTemp[pLev][1::2, 1::2, ::2] +
				   iTemp[pLev][::2, 1::2, ::2] + iTemp[pLev][1::2, ::2, 1::2] +
				   iTemp[pLev][1::2, ::2, ::2] + iTemp[pLev][::2, 1::2, 1::2])*0.125


# Solves at coarsest level using an iterative solver
def solve():
	global N, vLev
	global gsFactor
	global maxCount
	global pData, rData
	global hyhz, hzhx, hxhy, hxhyhz

	n = N[vLev]
	solLap = np.zeros(n)

	jCnt = 0
	while True:
		imposePpBCs(pData[vLev])

		
		pData[vLev][1:n[0]+1, 1:n[1]+1, 1:n[2]+1] = (hyhz[vLev]*(pData[vLev][2:n[0]+2, 1:n[1]+1, 1:n[2]+1] + pData[vLev][0:n[0], 1:n[1]+1, 1:n[2]+1]) +
													 hzhx[vLev]*(pData[vLev][1:n[0]+1, 2:n[1]+2, 1:n[2]+1] + pData[vLev][1:n[0]+1, 0:n[1], 1:n[2]+1]) +
													 hxhy[vLev]*(pData[vLev][1:n[0]+1, 1:n[1]+1, 2:n[2]+2] + pData[vLev][1:n[0]+1, 1:n[1]+1, 0:n[2]]) -
													 hxhyhz[vLev]*rData[vLev][0:n[0], 0:n[1], 0:n[2]]) * gsFactor[vLev]

		'''
		# Gauss-Seidel iterative solver
		for i in range(1, n[0]+1):
			for j in range(1, n[1]+1):
				for k in range(1, n[2]+1):
					pData[vLev][i, j, k] = (hyhz[vLev]*(pData[vLev][i+1, j, k] + pData[vLev][i-1, j, k]) +
											hzhx[vLev]*(pData[vLev][i, j+1, k] + pData[vLev][i, j-1, k]) +
											hxhy[vLev]*(pData[vLev][i, j, k+1] + pData[vLev][i, j, k-1]) -
										  hxhyhz[vLev]*rData[vLev][i-1, j-1, k-1]) * gsFactor[vLev]
		'''

		maxErr = np.amax(np.abs(rData[vLev] - laplace(pData[vLev])))
		if maxErr < PoissonTolerance:
			#print(vLev, jCnt)
			break

		jCnt += 1
		if jCnt > maxCount:
			print("ERROR: Poisson solver not converging. Aborting")
			quit()

	imposePpBCs(pData[vLev])


# Increases the size of the array to a higher level, 2^(n + 1) + 1
def prolong():
	global N
	global vLev
	global pData

	pLev = vLev
	vLev -= 1

	pData[vLev].fill(0.0)

	n = N[vLev]
	for i in range(1, n[0] + 1):
		i2 = int((i-1)/2) + 1
		for j in range(1, n[1] + 1):
			j2 = int((j-1)/2) + 1
			for k in range(1, n[2] + 1):
				k2 = int((k-1)/2) + 1
				pData[vLev][i, j, k] = pData[pLev][i2, j2, k2]


# Computes the 3D laplacian of function
def laplace(function):
	global N, vLev
	global mghx2, mghy2, mghz2

	n = N[vLev]

	laplacian = ((function[:-2, 1:-1, 1:-1] - 2.0*function[1:-1, 1:-1, 1:-1] + function[2:, 1:-1, 1:-1])/mghx2[vLev] + 
				 (function[1:-1, :-2, 1:-1] - 2.0*function[1:-1, 1:-1, 1:-1] + function[1:-1, 2:, 1:-1])/mghy2[vLev] +
				 (function[1:-1, 1:-1, :-2] - 2.0*function[1:-1, 1:-1, 1:-1] + function[1:-1, 1:-1, 2:])/mghz2[vLev])

	return laplacian


# Multigrid V-cycle without the use of recursion
def v_cycle():
	global VDepth
	global vLev
	global pstSm, preSm

	vLev = 0

	# Pre-smoothing
	smooth(preSm)

	for i in range(VDepth):
		# Compute residual
		calcResidual()

		# Copy smoothed pressure for later use
		sData[vLev] = np.copy(pData[vLev])

		# Restrict to coarser level
		restrict()

		# Reinitialize pressure at coarser level to 0 - this is critical!
		pData[vLev].fill(0.0)

		# If the coarsest level is reached, solve. Otherwise, keep smoothing!
		if vLev == VDepth:
			solve()
			
			#smooth(preSm)
		else:
			smooth(preSm)

	# Prolongation operations
	for i in range(VDepth):
		# Prolong pressure to next finer level
		prolong()

		# Add previously stored smoothed data
		pData[vLev] += sData[vLev]

		smooth(pstSm)


# The root function of MG-solver. And H is the RHS
def Poisson_MG(H):
	global N
	global vcCnt
	global rConv
	global pAnlt
	global pData, rData

	rData[0] = H[1:-1, 1:-1, 1:-1]
	chMat = np.zeros(N[0])

	#for i in range(vcCnt):
	vcCnt = 0
	while True:

		v_cycle()

		if vcCnt > 100:
			print("Poisson solver not converging")
			quit()

		vcCnt += 1

		chMat = laplace(pData[0])
		resVal = np.amax(np.abs(H[1:-1, 1:-1, 1:-1] - chMat))

		#print("Residual after V-Cycle {0:2d} is {1:.4e}".format(vcCnt, resVal))

		if resVal < PoissonTolerance:
			#print(vcCnt)
			break

	return pData[0]


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

    #T[1:Nx-1, 1:Ny-1, 1:Nz-1] = (1.0 - 0.5/(Nz-1)) - z[1:Nz-1]

    #print(T[5, 5, 1:Nz-1])

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


def writeSoln(U, V, W, P, T, time):

    fName = "Soln_" + "{0:09.5f}.h5".format(time)
    #print("#Writing solution file: ", fName)        
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
    #print("#Writing Restart file: ", fName)        
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



def Poisson_Jacobi(rho):   
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


iCnt = 1
while True:

    t1 = datetime.now()

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

    #Pp = Poisson_Jacobi(rhs)
    Pp = Poisson_MG(rhs)

    tp2 = datetime.now()
    #print(tp2-tp1)    

    P = P + Pp

    #imposePpBCs(Pp)

    U[1:Nx-1, 1:Ny-1, 1:Nz-1] = U[1:Nx-1, 1:Ny-1, 1:Nz-1] - dt*(Pp[2:Nx, 1:Ny-1, 1:Nz-1] - Pp[0:Nx-2, 1:Ny-1, 1:Nz-1])/(2.0*hx)
    V[1:Nx-1, 1:Ny-1, 1:Nz-1] = V[1:Nx-1, 1:Ny-1, 1:Nz-1] - dt*(Pp[1:Nx-1, 2:Ny, 1:Nz-1] - Pp[1:Nx-1, 0:Ny-2, 1:Nz-1])/(2.0*hy)
    W[1:Nx-1, 1:Ny-1, 1:Nz-1] = W[1:Nx-1, 1:Ny-1, 1:Nz-1] - dt*(Pp[1:Nx-1, 1:Ny-1, 2:Nz] - Pp[1:Nx-1, 1:Ny-1, 0:Nz-2])/(2.0*hz)


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
        if iCnt == 1:
            print('# time \t\t Re \t\t Nu \t\t Divergence')
        print("%f \t %.8f \t %.8f \t %.8f" %(time, Re, Nu, maxDiv))           


    if abs(rwTime - time) < 0.5*dt:
        rwTime = rwTime + rwInt
        if iCnt > 1:
            writeRestart(U, V, W, P, T, time)
        

    if abs(fwTime - time) < 0.5*dt:
        writeSoln(U, V, W, P, T, time)
        fwTime = fwTime + fwInt  

    '''
    if abs(time - tMax)<1e-5:
        Z, Y = np.meshgrid(y,z)
        plt.contourf(Y, Z, T[int(Nx/2), :, :], 500, cmap=cm.coolwarm)
        clb = plt.colorbar()
        plt.quiver(Y[0:Nx,0:Ny], Z[0:Nx,0:Ny], V[int(Nx/2),:, :], W[int(Nx/2), :, :])
        plt.axis('scaled')
        clb.ax.set_title(r'$T$', fontsize = 20)
        plt.show()
    '''
 
    if abs(time - tMax) < 0.5*dt:
        print("Simulation completed!")
        break   

    time = time + dt

    iCnt = iCnt + 1

    t2 = datetime.now()


    #print(t2-t1)










