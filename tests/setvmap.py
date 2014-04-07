# -*- coding: utf-8 -*-

from hotspin import *
from math import *

eps = 1.0e-6

Nx = 128
Ny = 32
Nz = 1
setgridsize(Nx, Ny, Nz)
setcellsize(500e-9 / Nx, 125e-9 / Ny, 3e-9 / Nz)


X = [0 for x in xrange(400)]
Y = [[0 for x in range(3)] for x in xrange(400)]

load('llbar')
load('zeeman')

add_to('H_eff', 'H_ext1')

for i in range(400):
    X[i] = float(i)
    Y[i][0] = float(i + 1)
    Y[i][1] = float(i + 2)
    Y[i][2] = float(i + 3)
setvmap("H_ext1", "t", Y, X)


error = 0
for i in range(400):
    setv('t', float(i))
    HX = float(i + 1)
    HY = float(i + 2)
    HZ = float(i + 3)
    H1 = getv("H_ext1")
    error += abs(H1[0] - HX)
    error += abs(H1[1] - HY)
    error += abs(H1[2] - HZ)

verror = error / (Nx * Ny * Nz)

print('Absolute error per cell:', verror)

if verror > eps:
    print("\033[31m" + "✘ FAILED" + "\033[0m")

    sys.exit(1)
else:
    print("\033[32m" + "✔ PASSED" + "\033[0m")
    sys.exit()

printstats()
