# -*- coding: utf-8 -*-

from math import *
from hotspin import *


Nx = 512
Ny = 512
Nz = 1

sX = 512e-9
sY = 512e-9
sZ = 15e-9

csX = sX / Nx
csY = sY / Ny
csZ = sZ / Nz

setgridsize(Nx, Ny, Nz)
setcellsize(csX, csY, csZ)
setperiodic(1, 1, 0)

load('temperature/ETM')

load('temperature/ETM/diffusion')

add_to('Qe', 'Qe_spat')

load('solver/rk12')
setv('Te_maxerror', 1e-5)

savegraph("deps.png")

T = makearray(1, Nx, Ny, Nz)

for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
            val = 10.0
            if (i > 128) and (i < 384):
                if (j > 128) and (j < 384):
                    val = 300.0
            T[0][i][j][k] = val
setarray('Te', T)

save('Te', "dump", [])

setv('Ce', 1.0e6)

setv('Ke', 91.0)

autosave("Te", "dump", [], 1e-12)

autotabulate(["t", "<Te>"], "Te.dat", 1e-12)

setv('dt', 1e-18)

setv('mindt', 1e-22)

run(1.0e-11)

printstats()

sync()
