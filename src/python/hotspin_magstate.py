## @package mumax2_magstate
# Functions to generate arrays representing various magnetization states
# @author Arne Vansteenkiste

from mumax2 import *

def vortex(c,p):
	Nx,Ny,Nz = getgridsize()
	Cx,Cy,Cz = getcellsize()
	m= [[[[0 for x in range(Nz)] for x in range(Ny)] for x in range(Nx)] for comp in range(3)]

	# set circulation
	for i in range(0,Nx):
			x = ((-Nx/2. + i + 0.5))
			for j in range(0,Ny):
				y = ((-Ny/2. + j + 0.5))
				for k in range(0,Nz):
					z = ((-Nz/2. + k + 0.5))
					m[0][i][j][k] = -y*c
					m[1][i][j][k] = x*c
					m[2][i][j][k] = 0

	# set core
	for k in range(0,Nz):
		m[2][Nx/2][Ny/2][k] = p

	return m


