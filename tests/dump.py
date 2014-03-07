# -*- coding: utf-8 -*-

from hotspin import *
from math import *

# Standard Problem 4

eps = 1.0e-6

Nx = 128
Ny = 32
Nz = 1
setgridsize(Nx, Ny, Nz)
setcellsize(500e-9/Nx, 125e-9/Ny, 3e-9/Nz)

load('llbar')

mx = 1.0
my = 1.0
mz = 0.0

m=[ [[[mx]]], [[[my]]], [[[mz]]] ]
setarray('mf', m)
saveas("mf", "dump", [], "m.dump")

m=[ [[[0]]], [[[0]]], [[[0]]] ]
setarray('mf', m)

setarray_file('mf', outputdirectory()+"/m.dump")

m = getarray("mf")

error = 0.0
for k in range(Nz):
    for j in range(Ny):
        for i in range(Nx):
        	error += abs(m[0][i][j][k] - mx)
        	error += abs(m[1][i][j][k] - my)
        	error += abs(m[2][i][j][k] - mz)

verror =  error / (Nx * Ny * Nz) 

print "Absolute error per cell:", verror

if verror > eps:
    print "\033[31m" + "✘ FAILED" + "\033[0m"
    
    sys.exit(1)
else:
    print "\033[32m" + "✔ PASSED" + "\033[0m"
    sys.exit()

printstats()
