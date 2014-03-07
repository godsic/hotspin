# -*- coding: utf-8 -*-

from mumax2 import *
from math import *

Nx = 1
Ny = 1
Nz = 1

sX = 320e-9
sY = 320e-9
sZ = 20e-9

hsX = 0.5 * sX
hsY = 0.5 * sY
hsZ = 0.5 * sZ

csX = sX/Nx
csY = sY/Ny
csZ = sZ/Nz

setgridsize(Nx, Ny, Nz)
setcellsize(csX, csY, csZ)

# LLB

load('temperature/Debye')

savegraph("graph.png")

Ms0 = 516967
J = 0.5
g = 2.19
g_LL = g * mu0 * muB * 2.0 * pi / 6.62606896e-34
print g_LL
ro = Ms0 / (g * muB * J)
Tc= 633

setv('Td', 400.0)
setv('n', ro)

Ta=10.0
Tb=700.0
steps=1000
dT = (Tb-Ta)/steps

savegraph("graph.png")

C = [[[[1.0]]]]
setmask('Cp_l', C)

f = open('Cp.dat', 'w')

for i in range(steps):

    Temp = Ta + float(i)*dT
    T = [[[[Temp]]]]
    setarray('Tl', T)
    cp = getcell("Cp_l", 0, 0, 0)[0]
    print cp
    out = str(Temp) + '\t' + str(cp) + '\n'
    f.write(out)

f.flush()
f.close()

printstats()

sync()
