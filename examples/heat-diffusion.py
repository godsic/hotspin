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

# set mesh size
setgridsize(Nx, Ny, Nz)
# set cell size
setcellsize(csX, csY, csZ)
# enable PBC in-plane
setperiodic(1, 1, 0)

# load heat equation for electrons
load('temperature/ETM')
# load heat diffusion term for electrons
load('temperature/ETM/diffusion')

# add heat diffusion term to electron' heat equation
add_to('Qe', 'Qe_spat')

# use Adams-Moulton solver as it is way more stable than Runge-Kutta
load('solver/am12')
setv('Te_maxabserror', 1e-5)
setv('Te_maxrelerror', 1e-5)

savegraph("deps.png")

# setup initial profile of the temperature
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

# save initial temperature profile
save('Te', "dump", [])

# set volume-specific heat capacity of electrons, [J/(K*m^3)]
setv('Ce', 1.0e6)
# set heat conductivity of electrons, [W/(K*m)]
setv('Ke', 91.0)

# schedule output
autosave("Te", "dump", [], 1e-12)
autotabulate(["t", "<Te>"], "Te.dat", 1e-12)

# set initial time step
setv('dt', 1e-18)

# adjust solver's time step bounds
setv('mindt', 1e-22)

run(1.0e-11)

# print runtime statistics
printstats()

sync()
