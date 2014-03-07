# -*- coding: utf-8 -*-

from hotspin import *
from random import *
from math import *
# Standard Problem 4

# define geometry

eps = 1.0e-6

# number of cells
Nx = 64
Ny = 64
Nz = 64
setgridsize(Nx, Ny, Nz)

# physical size in meters
sizeX = 320e-9
sizeY = 160e-9
sizeZ = 64e-9
setcellsize(sizeX/Nx, sizeY/Ny, sizeZ/Nz)

seed(0)

# load modules
load('llbar')
load('demag')

# set parameters
msk=makearray(1, Nx, Ny, Nz)
for k in range(Nz):
    for j in range(Ny):
        for i in range(Nx):
            msk[0][i][j][k] = random() 
setmask('Msat0T0', msk)
setv('Msat0T0', 800e3)

# set magnetization
m=makearray(3, Nx, Ny, Nz)
for k in range(Nz):
    for j in range(Ny):
        for i in range(Nx):
            mx = random() 
            my = random()
            mz = random()
            l = sqrt(mx**2 + my**2 + mz**2)
            m[0][i][j][k] = mx / l 
            m[1][i][j][k] = my / l
            m[2][i][j][k] = mz / l
setarray('mf', m)

saveas('B', "dump", [], "B_new.dump")
ref = readfile(outputdirectory()+"/../B_ref.dump")
new = getarray('B')

error = 0.0

for k in range(Nz):
    for j in range(Ny):
        for i in range(Nx):
            for c in range(3):
               error += abs(new[c][i][j][k] - ref[c][i][j][k])
verror =  error / (Nx * Ny * Nz)

print "Absolute error per cell:", verror

if verror > eps:
    print "\033[31m" + "✘ FAILED" + "\033[0m"
    sys.exit(1)
else:
    print "\033[32m" + "✔ PASSED" + "\033[0m"
    sys.exit()
