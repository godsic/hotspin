# -*- coding: utf-8 -*-
from hotspin import *
from math import *

# Geometry
Nx = 128
Ny = 128
Nz = 16

sX = 640e-9
sY = 640e-9
sZ = 80e-9

hsX = 0.5 * sX
hsY = 0.5 * sY
hsZ = 0.5 * sZ

csX = sX / Nx
csY = sY / Ny
csZ = sZ / Nz

# set mesh size
setgridsize(Nx, Ny, Nz)
# set cell size
setcellsize(csX, csY, csZ)

# load unifrom exchange field according to MFA
loadargs('mfa/longfield',
         ["T=Te"],
         [],
         [])
# load 6-neighbors exchange module
load('exchange6')
# load demagnetization field module
load('demag')

# add uniform exchange field to the internal field
add_to('H_eff', 'H_lf')

# load LHS of the LLBar equation
load('llbar')
# load torque
load('llbar/torque')
# load zero-order relativistic relaxation term
load('llbar/damping/nonconservative/00/local')

# add torque to RHS of LLBar equation
add_to('llbar_RHS', 'llbar_torque')
# add zero-order local relativistic relaxation term to RHS of LLBar equation
add_to('llbar_RHS', 'llbar_local00nc')

# use Adams-Moulton solver as it is way more stable than Runge-Kutta
load('solver/am12')

savegraph("graph.png")

# set material parameters

Ms0 = 516967

g = 2.19
g_LL = g * mu0 * muB * 2.0 * pi / 6.62606896e-34
J = 0.5
ro = Ms0 / (g * muB * J)
v = 6.0221415e23 / ro
Tc = 633.0
lex = 5.8e-9
T = 1000.0
TT = [[[[T]]]]
lbd = 0.013

# set temperature of electrons, [K]
setarray('Te', TT)
# set Curie temperature, [K]
setv('Tc', Tc)
# set atomic spin, []
setv('J', J)
# set number density of spins, [1/m^3]
setv('n', ro)
# set magnetization saturation at zero temperature, [A/m]
setv('msat0T0', Ms0)
# set exchange length, [m^2]
setv('lex', lex)
# set Landau gyromagnetic ratio, [m/(As)]
setv('γ_LL', g_LL)
# set zero-order relativistic relaxation tensor (diagonal), []
setv('α', [lbd, lbd, lbd])

# make sure the storage is allocated for the mask-type quantities
msat = [[[[1.0]]]]
setmask('msat0T0', msat)

# set magnetization length to the equilibrium value at the given temperature
m = [[[[0.0]]], [[[0.0]]], [[[0.0]]]]
setarray('m', m)

# adjust solver's time step bounds
setv('maxdt', 1e-12)
setv('mindt', 1e-36)
# set initial time step
setv('dt', 1e-18)

# relax at Te=1000 K without thermal fluctuations
run(5.0e-12)

# load stochastic field according to Brown
loadargs('temperature/brown',
         [],
         ["T=Te", "mu=α"],
         [])

# add stochastic field to the internal field
add_to('H_eff', 'H_therm')
# set minimum correlation time of the electrons, [s]
setv('cutoff_dt', 1.0e-14)
# set seed for the random field used by the 'temperature/brown' module, []
setv('therm_seed', 1)

# reset time
setv('t', 0)
# set initial time step
setv('dt', 1e-18)

# schedule output
tt = 1.0e-14
autotabulate(["t", "<m>"], "m.dat", tt)
autosave("m", "dump", [], tt)

# relax at Te=1000 K with thermal fluctuations
run(50e-12)

# set initial time step
setv('dt', 1e-18)

T = 300.0
TT = [[[[T]]]]
# set temperature of electrons, [K]
setarray('Te', TT)

# relax at Te=300 K with thermal fluctuations
run(50e-9)

# print runtime statistics
printstats()

sync()
