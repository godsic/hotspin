## @package mumax2_geom
# Functions to generate masks representing geometrical shapes
# @author Arne Vansteenkiste

import math
from mumax2_cmp import *
import sys
from mumax2 import *

## Returns an array with ones inside an ellipsoid with semi-axes rx,ry,rz (in meters)
#  and 0 outside. Typically used as a mask.
def ellipsoid(rx, ry, rz):
	Nx,Ny,Nz = getgridsize()
	Cx,Cy,Cz = getcellsize()
	mask= [[[0 for x in range(Nz)] for x in range(Ny)] for x in range(Nx)]
	for i in range(0,Nx):
			x = ((-Nx/2. + i + 0.5) * Cx)
			for j in range(0,Ny):
				y = ((-Ny/2. + j + 0.5) * Cy)
				for k in range(0,Nz):
					z = ((-Nz/2. + k + 0.5) * Cz)
					if (x/rx)**2 + (y/ry)**2 + (z/rz)**2 < 1:
							mask[i][j][k] = 1
					else:
							mask[i][j][k] = 0
	return [mask]



## Returns a mask for a (2D) ellipse that fits exactly in the simulation box.
def ellipse():
	Nx,Ny,Nz = getgridsize()
	mask= [[[0 for x in range(Nz)] for x in range(Ny)] for x in range(Nx)]
	rx = Nx/2.
	ry = Ny/2.
	for i in range(0,Nx):
			x = (-Nx/2. + i + 0.5)
			for j in range(0,Ny):
				y = (-Ny/2. + j + 0.5)
				for k in range(0,Nz):
					if (x/rx)**2 + (y/ry)**2 < 1:
							mask[i][j][k] = 1
					else:
							mask[i][j][k] = 0
	return [mask]
	
## Returns a 3-component mask for a (2D) ellipse that fits exactly in the simulation box.
def ellipsevec(comp, vec):
	Nx,Ny,Nz = getgridsize()
	mask= [[[[0 for z in range(Nz)] for y in range(Ny)] for x in range(Nx)] for c in range(comp)]
	rx = Nx/2.
	ry = Ny/2.
	zval = [0 for c in range(comp)]
	for i in range(0,Nx):
			x = (-Nx/2. + i + 0.5)
			for j in range(0,Ny):
				y = (-Ny/2. + j + 0.5)
				for k in range(0,Nz):
					temp = zval
					if (x/rx)**2 + (y/ry)**2 < 1:
						temp = vec
					for c in range(comp):
						mask[c][i][j][k] = temp[c]
	return mask

## Returns a mask for a (2D) regulqr Ngone given a number of sides, a radius, a rotation angle and a center.
# @param sides (int) the number of sides (e.g. 3 for triangle, 4 for square ...) (unitless)
# @param radius (float) the radius of the circumscribed circle of the Ngone (m)
# @param rotation (float) the rotation angle applied to the Ngone (center being the center of the circumscribed circle) (degree)
# @param centerX (float) the X coordinate of the center of the circumscribed circle of the Ngone (m)
# @param centerY (float) the Y coordinate of the center of the circumscribed circle of the Ngone (m)
# @param Zmin (float) the Z coordinate of the lower side of the solid (m)
# @param Zmax (float) the Z coordinate of the higher side of the solid (m)
# @param region (string) name of the region to assign to this Ngone (if zero, the region system will not be used)
def Ngone(sides, radius, rotation, centerX, centerY, Zmin, Zmax, region = 0):
	load('regions')
	#mask= [[[0 for x in range(Nz)] for x in range(Ny)] for x in range(Nx)]
	setupRegionSystem()
	regionDefinition = getRegionDefinition()
	regionNameDictionary = getRegionNameDictionary()
	#add region to the list of region if not already included
	for i in range(0,len(regionNameDictionary)):
		if regionNameDictionary.has_key(region):
			break
		elif i == len(regionNameDictionary) -1:
			regionNameDictionary[region] = float(len(regionNameDictionary))
	setRegionNameDictionary(regionNameDictionary)
	Nx,Ny,Nz = getgridsize()
	Cx,Cy,Cz = getcellsize()
	diagonal =  radius * math.cos(math.pi / sides)
	rotRad = rotation * math.pi / 180.
	if Zmin > Zmax :
		tmp = Zmin
		Zmin = Zmax
		Zmax = tmp
	for i in range(0,Nx):
		x = (i + 0.5)*Cx - centerX
		for j in range(0,Ny):
			y = (j + 0.5)*Cy - centerY
			theta = math.atan2(y, x)
			if theta < rotRad:
				theta += 2 *math.pi
			for n in range(0,sides):
				if theta <= 2*math.pi*(n+1)/sides + rotRad and theta > 2*math.pi*n/sides + rotRad:
					angle = -2*math.pi*(n+0.5)/sides - rotRad
					xx = x * math.cos(angle) - y * math.sin(angle)
					if xx < diagonal:
						for k in range(0,Nz):
							z = (k+0.5)*Cz
							if z >= Zmin and z <= Zmax:
								regionDefinition[i][j][k] = regionNameDictionary[region]
	setmask('regionDefinition', [regionDefinition])
	setRegionDefinition(regionDefinition)
	"""	Nx,Ny,Nz = getgridsize()
	Cx,Cy,Cz = getcellsize()
	mask= [[[0 for x in range(Nz)] for x in range(Ny)] for x in range(Nx)]
	if region > 0:
		setupRegionSystem()
		regionDefinition = getarray('regionDefinition')
	for i in range(0,Nx):
		x = (i + 0.5)*Cx - centerX
		for j in range(0,Ny):
			y = (j + 0.5)*Cy - centerY
			# calculate polar angle
			theta = math.atan2(y, x)
			for n in range(0,sides):
				if theta <= 2*math.pi*(n+1)/sides and theta > 2*math.pi*n/sides:
					angle = -2*math.pi*(n+0.5)/sides + rotation * math.pi / 180.
					xx = x * math.cos(angle) - y * math.sin(angle)
					if xx < radius * math.cos(math.pi / sides):
						for k in range(0,Nz):
							mask[i][j][k] = 1
							if region > 0:
								regionDefinition[i][j][k] = 1
					else:
						for k in range(0,Nz):
							mask[i][j][k] = 0
					break
	if region > 0:
		setmask('regionDefinition', regionDefinition)
	return [mask]
	"""
	
