# -*- coding: utf-8 -*-

from math import *
from hotspin import *

Nx = 32
Ny = 32
Nz = 4

sX = 160e-9
sY = 160e-9
sZ = 15e-9

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

# enable PBC in-plane
setperiodic(8, 8, 0)

# load 6-neighbors exchange module
load('exchange6')

# load heat equation for electrons
load('temperature/ETM')

# load heat equation for phonons
load('temperature/PTM')

# load electron-phonon coupling module
load('temperature/E-P')

# load Debye model for the phonon' heat capacity
load('temperature/Debye')

# load Drude model for the electron' heat capacity
load('temperature/Drude')

# load unifrom exchange field according to MFA
loadargs('mfa/longfield', ["T=Te"], [], ["H_lf=H_lf_e"])

# load temperature dependence of the equilibrium magnetization length
# according to MFA
loadargs('mfa/msat0', ["T=Te"], [], [])

# add uniform exchange field to the internal field
add_to('H_eff', 'H_lf_e')

# load LHS of the LLBar equation
load('llbar')

# load zero-order relativistic relaxation term
loadargs('llbar/damping/nonconservative/00/local',
         [],
         [],
         ["llbar_local00nc=llbar_local00nc"])

# load internal energy dissipation rate
loadargs('mfa/molecular-energy-density-rate',
         [],
         ["R=llbar_local00nc"],
         ["w=ws00"])

# load entropy production rate
loadargs('energy-density-dissipation-rate',
         [],
         ["R=llbar_local00nc"],
         ["q=qs00"])

# add zero-order local relativistic relaxation term to RHS of LLBar equation
add_to('llbar_RHS', 'llbar_local00nc')

# add spin-electron coupling term to the electron' heat equation
add_to_weighted("Qe", "ws00", 1.0)
add_to_weighted("Qe", "qs00", -1.0)

# add laser heat source to the electron' heat equation
add_to("Qe", "Qlaser")

# add electron-coupling term to the electron' and phonon' heat equation
add_to_weighted("Qe", "Qe-p", 1.0)
add_to_weighted("Qp", "Qe-p", -1.0)

# use Adams-Moulton solver as it is way more stable than Runge-Kutta
load('solver/am12')
setv('m_maxabserror', 1e-5)
setv('m_maxrelerror', 1e-5)
setv('Te_maxabserror', 1e-5)
setv('Te_maxrelerror', 1e-5)
setv('Tp_maxabserror', 1e-5)
setv('Tp_maxrelerror', 1e-5)

savegraph("deps.png")

# set material parameters
Ms0 = 516967

T = [[[[200.0]]]]
# set temperature of electrons, [K]
setarray('Te', T)
# set temperature of phonons, [K]
setarray('Tp', T)

g = 2.19
g_LL = g * mu0 * muB * 2.0 * pi / 6.62606896e-34
J = 0.5
ro = Ms0 / (g * muB * J)
v = 6.0221415e23 / ro
Tc = 633.0
Td = 390.0
Aex = 0.86e-11
lex = sqrt(2.0 * Aex / (mu0 * Ms0 * Ms0))
lbd = 1.37232753598e-2

echo(str(g_LL))
echo(str(v))

# set electron' specific heat constant, [J/(K^2 mol)]
setv('γ', 0.0045070746312099)
# set electron-phonon coupling constant, [W/(K m^3)]
setv('Ge-p', 1.35155966523e+18)
# set Curie temperature, [K]
setv('Tc', Tc)
# set number density of spins, [1/m^3]
setv('n', ro)
# set atomic spin, []
setv('J', J)
# set Debye temperature, [K]
setv('Td', Td)
# set exchange length, [m^2]
setv('lex', lex)
# set Landau gyromagnetic ratio, [m/(As)]
setv('γ_LL', g_LL)
# set zero-order relativistic relaxation tensor (diagonal), []
setv('α', [lbd, lbd, lbd])
# set multiplied of the magnetization saturation as it is required by 'mfa/msat0' module, [A/m]
setv('msat0', Ms0)
# set magnetization saturation at zero temperature, [A/m]
setv('msat0T0', Ms0)

# make sure the storage is allocated for the mask-type quantities
msat = [[[[1.0]]]]
setmask('msat0', msat)
setmask('msat0T0', msat)
setmask('Cp', msat)
setmask('Ce', msat)

# set magnetization length to the equilibrium value at the given temperature
Ms = getcell("msat0", 15, 15, 1)[0]
m = [[[[Ms / Ms0]]], [[[0.0]]], [[[0.0]]]]
setarray('m', m)

echo(str(Ms))


# setup laser heating
tt = 1e-15
T0 = 500e-15  # Time delay for the excitation
FWHM = 50e-15
dT = FWHM / (2.0 * sqrt(2.0 * log(2.0)))
dTT = 0.5 / (dT * dT)  # FWHW squared

N = 3100
time = N * tt
fine = 100
N_fine = fine * N
tt_fine = tt / float(fine)

Qamp = 8.90302203155e+21

T = [0 for x in xrange(N_fine + 1)]
Q = [[0 for x in range(1)] for x in xrange(N_fine + 1)]

for i in range(N_fine):
        t = tt_fine * float(i)
        T[i] = t
        Q[i][0] = Qamp * exp(-1.0 * dTT * (t - T0) ** 2)

T[N_fine] = 9999.9
Q[N_fine][0] = 0.0
setvmap('Qlaser', 't', Q, T)


# schedule output
autotabulate(["t", "<m>"], "m.txt", tt)
autotabulate(["t", "<Tp>"], "Tp.txt", tt)
autotabulate(["t", "<Te>"], "Te.txt", tt)
autotabulate(["t", "<Ce>"], "Ce.txt", tt)
autotabulate(["t", "<Cp>"], "Cp.txt", tt)
autotabulate(["t", "<Qe-p>"], "qep.txt", tt)

# adjust solver's time step bounds
setv('maxdt', 1e-14)
setv('mindt', 1e-22)

# set initial time step
setv('dt', 1e-17)

run(5e-12)

setv('maxdt', 1e-12)
run(20e-12)

# print runtime statistics
printstats()

sync()
