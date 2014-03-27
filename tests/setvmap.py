from hotspin import *
from math import *

Nx = 128
Ny = 32
Nz = 1
setgridsize(Nx, Ny, Nz)
setcellsize(500e-9/Nx, 125e-9/Ny, 3e-9/Nz)


X = [0 for x in xrange(400)]
Y = [[0 for x in range(3)] for x in xrange(400)]

load('llbar')
load('zeeman')

add_to('H_eff', 'H_ext1')
add_to('H_eff', 'H_ext2')

for i in range(400):
        t = float(i)
        setpointwise('H_ext1', t, [i + 1, i + 2, i + 3])

for i in range(400):
        X[i] = float(i)
     	Y[i][0] = float(i+1)
     	Y[i][1] = float(i+2)
     	Y[i][2] = float(i+3)
setvmap("H_ext2", "t", Y, X)


error = 0
for i in range(400):
		setv('t', float(i))
		H1 = getv("H_ext1")
		H2 = getv("H_ext2")
		print H1, H2
		error += abs(H1[0] - H2[0])
		error += abs(H1[1] - H2[1])
		error += abs(H1[2] - H2[2])

print error
