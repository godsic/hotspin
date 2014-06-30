#! /usr/bin/python
# -*- coding: utf-8 -*-
# @author: Mykola Dvornik
from hotspin import *
from math import *
# Standard Problem 4

Nx = 128
Ny = 32
Nz = 1
setgridsize(Nx, Ny, Nz)

sizeX = 500e-9
sizeY = 125e-9
sizeZ = 3e-9
setcellsize(sizeX / Nx, sizeY / Ny, sizeZ / Nz)

load('exchange6')
load('demag')
load('zeeman')

load('llbar')
load('llbar/torque')
load('llbar/damping/nonconservative/00/local')

add_to('llbar_RHS', 'llbar_torque')
add_to('llbar_RHS', 'llbar_local00nc')

load('normalizer')

loadargs('maxtorque', [],
         ["torque:llbar_torque"],
         [])

load('solver/rk12')
setv('m_maxerror', 1e-4)

setv('mindt', 1e-16)
setv('maxdt', 1e-12)

# set parameters
# Py


Mf = makearray(3, Nx, Ny, Nz)
for kk in range(Nz):
    for jj in range(Ny):
        for ii in range(Nx):
            Mf[0][ii][jj][kk] = 1.0 / sqrt(2.0)
            Mf[1][ii][jj][kk] = 1.0 / sqrt(2.0)
            Mf[2][ii][jj][kk] = 0.0
setarray('m', Mf)

msat0 = makearray(1, Nx, Ny, Nz)

for kk in range(Nz):
    for jj in range(Ny):
        for ii in range(Nx):
            msat0[0][ii][jj][kk] = 1.0

setmask('msat0', msat0)

Ms0 = 800e3
Aex = 1.3e-11
lex = sqrt(2 * Aex / (mu0 * Ms0 * Ms0))
setv('msat0', Ms0)
setv('msat0T0', Ms0)
setv('lex', lex)

alpha = 1.0
setv('α', [alpha, alpha, alpha])

gamma = 2.211e5
gammall = gamma / (1.0 + alpha ** 2)
setv('γ_ll', gammall)
setv('dt', 1e-17)  # initial time step, will adapt

# relax

autotabulate(["t", "maxtorque"], "t.dat", 1e-15)
# autotabulate(["t", "<mf>"], "mf-pre.txt", 1e-15)
run_until_smaller('maxtorque', 1e-3 * gammall * Ms0)

alpha = 0.02
gammall = gamma / (1.0 + alpha ** 2)
setv('γ_ll', gammall)
setv('α', [alpha, alpha, alpha])
setv('t', 0)        # re-set time to 0 so output starts at 0

autotabulate(["t", "<m>"], "m.txt", 1e-15)

# run with field

Bx = -24.6E-3
By = 4.3E-3
Bz = 0.0
setv('B_ext', [Bx, By, Bz])
setv('dt', 1e-15)  # initial time step, will adapt
run(1e-9)


# some debug output

printstats()

savegraph("graph.png")

sync()
