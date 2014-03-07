## @package mumax2_cmp
# This file contains function to set initial field given a array (with grid-size and cell-size already set up)

import os
import json
import png
import re
import sys
import math
from mumax2 import *


regionNameDictionary = {'empty':0.}
regionDefinition = [[]]
regionInitialised = False

##  Sets field (E.g. magnetization) as vortex (local or global) in the selected region.
# @param fieldName (string) the name of the field we should write to
# @param center (tuple 3 floats) the center of the vortex
# @param axis (tuple 3 floats) the direction of the core
# @param polarity (int) the sense of the core (-1 for against the axis or +1 for along the axis)
# @param chirality (int) the chirality of the vortex (-1 for CCW or +1 for CW)
# @param region (string) the region name that should contain the vortex. 'all' means all regions
# @param maxRadius (float) the maximum radius of the vortex. Could be used for vortex domain wall else use the default value 0. 
# @todo take into account Aex for setting the size of the core
# @todo use gaussian function for the core
# @todo add region support
def setVortex( fieldName , center , axis , polarity , chirality , region = 'all' , maxRadius = 0. ):
	## we assume that the engine return vectors as (z,y,x)
	## we assume that user will enter vector and center as (x,y,z)
	gridSize = getgridsize()
	cellSize = getcellsize()
	field = getarray(fieldName)
	setupRegionSystem()
	regionDefinition = getarray('regionDefinition')
	global regionNameDictionary
	axisNorm = math.sqrt(1/(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]))
	u = [axis[0] * axisNorm , axis[1] * axisNorm , axis[2] * axisNorm]
	for X in range(0,gridSize[0]):
		for Y in range(0,gridSize[1]):
			for Z in range(0,gridSize[2]):
				cellRegion = getcell('regionDefinition', X,Y,Z)
				#if ( region == 'all' and cellRegion != 0. ) or cellRegion == region:
				coordinateZ = ( float(Z) + 0.5 ) * cellSize[2]
				coordinateY = ( float(Y) + 0.5 ) * cellSize[1]
				coordinateX = ( float(X) + 0.5 ) * cellSize[0]	
				## component of v the shortest vector going from the line (passing by center and direction axis) and the current point
				v1 = [center[0] - coordinateX, center[1] - coordinateY, center[2] - coordinateZ]
				## scalar product v.u
				vScalaru = v1[0] * u[0] + v1[1] * u[1] + v1[2] * u[2]
				v = [0.,0.,0.]
				v[0] = v1[0] - u[0] * vScalaru
				v[1] = v1[1] - u[1] * vScalaru
				v[2] = v1[2] - u[2] * vScalaru
				## v norm
				d = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
				if maxRadius == 0. or d <= maxRadius:
					## set field to vortex
					if d < 2e-8:
						m = [ u[0] * polarity ,
							  u[1] * polarity ,
							  u[2] * polarity ]
						setcell(fieldName,X,Y,Z,m)
					else:
						m = [ - chirality * ( u[1] * v[2] - u[2] * v[1])/d ,
							  - chirality * ( u[2] * v[0] - u[0] * v[2])/d ,
							  - chirality * ( u[0] * v[1] - u[1] * v[0])/d ]
						setcell(fieldName,X,Y,Z,[float(m[0]),float(m[1]),float(m[2])])
	return

## Set up the region system
# Set up an array of the size of the grid filled with zeros and return it set up a dictionary of the regions name
def setupRegionSystem():
	global regionDefinition
	global regionNameDictionary
	global regionInitialised
	if not regionInitialised:
		load('regions')
		setv('regionDefinition',1.0)
		#build an array at the good size to prevent resample call
		gridSize = getgridsize()
		for i in range(0,gridSize[0]):
			regionDefinition[0].append([])
			for j in range(0,gridSize[1]):
				regionDefinition[0][i].append([])
				for k in range(0,gridSize[2]):
					regionDefinition[0][i][j].append(0.0)
		setmask('regionDefinition', regionDefinition)
		regionDefinition = getRegionDefinition()[0]
		regionInitialised = True
	return

def getRegionNameDictionary():
	global regionNameDictionary
	return regionNameDictionary

def setRegionNameDictionary(newDic):
	global regionNameDictionary
	regionNameDictionary = newDic
	return

def getRegionDefinition():
	global regionDefinition
	return regionDefinition

def setRegionDefinition(newDef):
	global regionDefinition
	regionDefinition = newDef
	return

## Set up and initialize the region system corresponding to a given script
# Set up regions given a script that return a region index for each voxel
# @param script (function) a function that will be called for each voxel. It should return a string to identify each region or 'empty' if the voxel does not correspond to any region. Its arguments should be three ints and a dictionary for any additional parameters (x, y, z, parameters)
# @param parameters (tuple) parameters to pass to the script 
def initRegionsScript( script , parameters):
	setupRegionSystem()
	global regionNameDictionary
	global regionDefinition
	regionDefinition = [[]]
	gridSize = getgridsize()
	regionNameList = regionNameDictionary.values()
	regionNameListLen = float(len(regionNameList))
	for i in range(0,gridSize[0]):
		regionDefinition[0].append([])
		for j in range(0,gridSize[1]):
			regionDefinition[0][i].append([])
			for k in range(0,gridSize[2]):
				result = script(i, j, k, parameters)
				regionDefinition[0][i][j].append(regionNameDictionary.get(result,regionNameListLen))
				if regionDefinition[0][i][j][k] == regionNameListLen:
					regionNameDictionary[result] = regionNameListLen
					regionNameListLen += 1.
	
	setmask('regionDefinition', regionDefinition)
	return

## Set up and initialize region system given a png imqge
# Set up regions by applying a png picture on a plane and extruding it perpendicularly to it
# @param imageName (string) is the name of the PNG image to use
# @param regionList (dictionary string (region name)=>string (color name or code)) associate a color to each region. The color could be either named if it is part of the <a href="http://www.w3schools.com/html/html_colornames.asp">standard html colors</a> or a string coding the color in the HTML hexadecimal format : "#XXXXXX" where X are between 0 and F
# @param plane (string) defines the plane to which the picture will be applied (default the plane xy). The first axis will be matched with the width of the image and the second axis with the height.
# @todo thickness (float) defines the extruded thickness. 0 means "through all".
# @todo origin (float) if thickness is not 0, then origin defines the starting point of the extrusion. It will happen along the increasing value of the extrusion axis.
# @note names of the colors allowed. @htmlinclude mumax2_cmp.py.html
def extrudeImage( imageName , regionList , plane = 'xy'):
	setupRegionSystem()
	global regionNameDictionary
	#test plane argument validity
	if not re.match('[x-z]{2}',plane,re.IGNORECASE) or plane[0] == plane[1]:
		print >> sys.stderr, 'extrudeImage plane cannot be %s' % plane
		sys.exit()
	htmlColorName = {
					'AliceBlue'		: '#F0F8FF',				'AntiqueWhite'	: '#FAEBD7',
					'Aqua'			: '#00FFFF',				'Aquamarine'	: '#7FFFD4',
					'Azure'			: '#F0FFFF',				'Beige'			: '#F5F5DC',
					'Bisque'		: '#FFE4C4',				'Black'			: '#000000',
					'BlanchedAlmond': '#FFEBCD',				'Blue'			: '#0000FF',
					'BlueViolet'	: '#8A2BE2',				'Brown'			: '#A52A2A',
					'BurlyWood'		: '#DEB887',				'CadetBlue'		: '#5F9EA0',
					'Chartreuse'	: '#7FFF00',				'Chocolate'		: '#D2691E',
					'Coral'			: '#FF7F50',				'CornflowerBlue': '#6495ED',
					'Cornsilk'		: '#FFF8DC',				'Crimson'		: '#DC143C',
					'Cyan'			: '#00FFFF',				'DarkBlue'		: '#00008B',
					'DarkCyan'		: '#008B8B',				'DarkGoldenRod'	: '#B8860B',
					'DarkGray'		: '#A9A9A9',				'DarkGrey'		: '#A9A9A9',
					'DarkGreen'		: '#006400',				'DarkKhaki'		: '#BDB76B',
					'DarkMagenta'	: '#8B008B',				'DarkOliveGreen': '#556B2F',
					'Darkorange'	: '#FF8C00',				'DarkOrchid'	: '#9932CC',
					'DarkRed'		: '#8B0000',				'DarkSalmon'	: '#E9967A',
					'DarkSeaGreen'	: '#8FBC8F',				'DarkSlateBlue'	: '#483D8B',
					'DarkSlateGray'	: '#2F4F4F',				'DarkSlateGrey'	: '#2F4F4F',
					'DarkTurquoise'	: '#00CED1',				'DarkViolet'	: '#9400D3',
					'DeepPink'		: '#FF1493',				'DeepSkyBlue'	: '#00BFFF',
					'DimGray'		: '#696969',				'DimGrey'		: '#696969',
					'DodgerBlue'	: '#1E90FF',				'FireBrick'		: '#B22222',
					'FloralWhite'	: '#FFFAF0',				'ForestGreen'	: '#228B22',
					'Fuchsia'		: '#FF00FF',				'Gainsboro'		: '#DCDCDC',
					'GhostWhite'	: '#F8F8FF',				'Gold'			: '#FFD700',
					'GoldenRod'		: '#DAA520',				'Gray'			: '#808080',
					'Grey'			: '#808080',				'Green'			: '#008000',
					'GreenYellow'	: '#ADFF2F',				'HoneyDew'		: '#F0FFF0',
					'HotPink'		: '#FF69B4',				'IndianRed'		: '#CD5C5C',
					'Indigo'		: '#4B0082',				'Ivory'			: '#FFFFF0',
					'Khaki'			: '#F0E68C',				'Lavender'		: '#E6E6FA',
					'LavenderBlush'	: '#FFF0F5',				'LawnGreen'		: '#7CFC00',
					'LemonChiffon'	: '#FFFACD',				'LightBlue'		: '#ADD8E6',
					'LightCoral'	: '#F08080',				'LightCyan'		: '#E0FFFF',
					'LightGoldenRodYellow'	: '#FAFAD2',		'LightGray'		: '#D3D3D3',
					'LightGrey'		: '#D3D3D3',				'LightGreen'	: '#90EE90',
					'LightPink'		: '#FFB6C1',				'LightSalmon'	: '#FFA07A',
					'LightSeaGreen'	: '#20B2AA',				'LightSkyBlue'	: '#87CEFA',
					'LightSlateGray': '#778899',				'LightSlateGrey': '#778899',
					'LightSteelBlue': '#B0C4DE',				'LightYellow'	: '#FFFFE0',
					'Lime'			: '#00FF00',				'LimeGreen'		: '#32CD32',
					'Linen'			: '#FAF0E6',				'Magenta'		: '#FF00FF',
					'Maroon'		: '#800000',				'MediumAquaMarine'	: '#66CDAA',
					'MediumBlue'	: '#0000CD',				'MediumOrchid'	: '#BA55D3',
					'MediumPurple'	: '#9370D8',				'MediumSeaGreen'	: '#3CB371',
					'MediumSlateBlue'	: '#7B68EE',			'MediumSpringGreen'	: '#00FA9A',
					'MediumTurquoise'	: '#48D1CC',			'MediumVioletRed'	: '#C71585',
					'MidnightBlue'	: '#191970',				'MintCream'		: '#F5FFFA',
					'MistyRose'		: '#FFE4E1',				'Moccasin'		: '#FFE4B5',
					'NavajoWhite'	: '#FFDEAD',				'Navy'			: '#000080',
					'OldLace'		: '#FDF5E6',				'Olive'			: '#808000',
					'OliveDrab'		: '#6B8E23',				'Orange'		: '#FFA500',
					'OrangeRed'		: '#FF4500',				'Orchid'		: '#DA70D6',
					'PaleGoldenRod'	: '#EEE8AA',				'PaleGreen'		: '#98FB98',
					'PaleTurquoise'	: '#AFEEEE',				'PaleVioletRed'	: '#D87093',
					'PapayaWhip'	: '#FFEFD5',				'PeachPuff'		: '#FFDAB9',
					'Peru'			: '#CD853F',				'Pink'			: '#FFC0CB',
					'Plum'			: '#DDA0DD',				'PowderBlue'	: '#B0E0E6',
					'Purple'		: '#800080',				'Red'			: '#FF0000',
					'RosyBrown'		: '#BC8F8F',				'RoyalBlue'		: '#4169E1',
					'SaddleBrown'	: '#8B4513',				'Salmon'		: '#FA8072',
					'SandyBrown'	: '#F4A460',				'SeaGreen'		: '#2E8B57',
					'SeaShell'		: '#FFF5EE',				'Sienna'		: '#A0522D',
					'Silver'		: '#C0C0C0',				'SkyBlue'		: '#87CEEB',
					'SlateBlue'		: '#6A5ACD',				'SlateGray'		: '#708090',
					'SlateGrey'		: '#708090',				'Snow'			: '#FFFAFA',
					'SpringGreen'	: '#00FF7F',				'SteelBlue'		: '#4682B4',
					'Tan'			: '#D2B48C',				'Teal'			: '#008080',
					'Thistle'		: '#D8BFD8',				'Tomato'		: '#FF6347',
					'Turquoise'		: '#40E0D0',				'Violet'		: '#EE82EE',
					'Wheat'			: '#F5DEB3',				'White'			: '#FFFFFF',
					'WhiteSmoke'	: '#F5F5F5',				'Yellow'		: '#FFFF00',
					'YellowGreen'	: '#9ACD32'
					}
	#first convert regionList to be fully coded in color hex code and fill regionNameDictionanry
	htmlCodeRE = re.compile('\#[a-f\d]{6}',re.IGNORECASE)
	colorToRegion = {}
	setupRegionSystem()
	regionNameListLen = float(len(regionNameDictionary))
	for i, cell in regionList.items():
		if cell[0] != '#':
			regionList[i] = htmlColorName[cell]
		if htmlCodeRE.match(regionList[i]):
			regionNameDictionary[i] = regionNameListLen
			colorToRegion[regionList[i]] = regionNameListLen
			regionNameListLen += 1.
	#Read picture
	imageReader = png.Reader( filename = imageName )
	imageWidth, imageHeight, pngData, meta = imageReader.read()
	rawRegionData = [[]]
	for rowIndex, row in enumerate(pngData):
		rawRegionData.append([])
		for columnIndex in range(0,imageWidth):
			pixel = [0,0,0,0]
			for comp in range(0,meta['planes']):
				pixel[comp] = row[columnIndex*meta['planes']+comp]
			colorCode = "#%02X%02X%02X" % (pixel[0],pixel[1],pixel[2])
			if colorCode in colorToRegion:
				rawRegionData[rowIndex].append(colorToRegion[colorCode])
			else:
				rawRegionData[rowIndex].append(regionNameDictionary['empty'])
	
	#Apply rawRegionData to regionDefinition mask by stretching in plane and extruding out of plane
	#rawRegionData
	gridSize = getgridsize()
	u = 0
	v = 0
	w = 0
	if plane[0] == 'x':
		u=0
	elif plane[0] == 'y':
		u=1
	elif plane[0] == 'z':
		u=2
	if plane[1] == 'x':
		v=0
	elif plane[1] == 'y':
		v=1
	elif plane[1] == 'z':
		v=2
	if u+v == 1:
		w = 2
	elif u+v == 2:
		w = 1
	else:
		w = 0
	global regionDefinition
	regionDefinition = [[]]
	for i in range(0,gridSize[u]):
		regionDefinition[0].append([])
		i1 = (i *imageWidth / gridSize[u])
		for j in range(0,gridSize[v]):
			regionDefinition[0][i].append([])
			j1 = imageHeight -1 - j * imageHeight / gridSize[v]
			for k in range(0,gridSize[w]):
				regionDefinition[0][i][j].append(rawRegionData[j1][i1])
	
	setmask('regionDefinition', regionDefinition)
	
	return

## Initialize scalar quantity with uniform value in each region
#  Every regions that have been set but that are not used here will be considered as part of 'empty' region. Empty region will set the quantity to zero.
# @param quantName (string) name of the scalar quantity to set
# @param initValues (dictionary string(region name) => float (value in region)) initial values. Any not existing region is ignored.
def InitUniformRegionScalarQuant(quantName, initValues):
	global regionNameDictionary
	Idx = 0.
	values = []
	for key, value in sorted(regionNameDictionary.iteritems(), key=lambda (k,v): (v,k)):
		if initValues.has_key(key):
			values.append(initValues[key])
		elif key == 'empty':
			values.append(0.)
		else:
			values.append(0.)
		Idx +=1
	setscalaruniformregion(quantName,values)
	return

## Initialize Vector quantity with uniform value in each region
#  Every regions that have been set but that are not used here will be considered as part of 'empty' region. Empty region will set the quantity to zero.
# @param quantName (string) name of the vector quantity to set
# @param initValues (dictionary string(region name) => [3]float (value in region)) initial values. Any not existing region is ignored.
def InitUniformRegionVectorQuant(quantName, initValues):
	global regionNameDictionary
	Idx = 0.
	valuesX = []
	valuesY = []
	valuesZ = []
	for key, value in sorted(regionNameDictionary.iteritems(), key=lambda (k,v): (v,k)):
		if initValues.has_key(key):
			valuesX.append(initValues[key][0])
			valuesY.append(initValues[key][1])
			valuesZ.append(initValues[key][2])
		elif key == 'empty':
			valuesX.append(0.1)
			valuesY.append(0.1)
			valuesZ.append(0.1)
		else:
			valuesX.append(0.1)
			valuesY.append(0.1)
			valuesZ.append(0.1)
		Idx +=1
	setvectoruniformregion(quantName, valuesX, valuesY, valuesZ )
	return

## Initialize scalar quantity with uniform value in each region
#  Every regions that have been set but that are not used here will be considered as part of 'empty' region. Empty region will set the quantity to zero.
# @param quantName (string) name of the scalar quantity to set
# @param initValues (dictionary string(region name) => float (value in region)) initial values. Any not existing region is ignored.
def InitVortexRegionVectorQuant(quantName, regionsToProceed, center, axis, polarity, chirality, maxRadius ):
	global regionNameDictionary
	regionsIndexToProceed = []
	cellSize = getcellsize()
	setupRegionSystem()
	if regionsToProceed.has_key('all') and regionsToProceed['all'] != 0.0:
		if len(regionNameDictionary) == 1:
			regionNameDictionary[1] = 'all'
		for key, value in sorted(regionNameDictionary.iteritems(), key=lambda (k,v): (v,k)):
			regionsIndexToProceed.append(1.0)
	else:
		for key, value in sorted(regionNameDictionary.iteritems(), key=lambda (k,v): (v,k)):
			if regionsToProceed.has_key(key):
				regionsIndexToProceed.append(regionsToProceed[key])
			elif key == 'empty':
				regionsIndexToProceed.append(0.0)
			else:
				regionsIndexToProceed.append(0.0)
	#setvectorvortexregion(quantName, regionsIndexToProceed, center, axis, [cellSize[0],cellSize[1],cellSize[2]], polarity, chirality, maxRadius )
	setvectorvortexregion(quantName, regionsIndexToProceed, center, axis, cellSize, polarity, chirality, maxRadius )
	return

## Initialize scalar quantity with uniform value in each region
#  Every regions that have been set but that are not used here will be considered as part of 'empty' region. Empty region will set the quantity to zero.
# @param quantName (string) name of the scalar quantity to set
# @param regionsToProceed (dictionary string(region name) => float (1 or 0)) 1 if region should be set up else 0.
# @param max (float) upper limit of the range of random number
# @param min (float) lower limit of the range of random number
def InitRandomUniformRegionScalarQuant(quantName, regionsToProceed, max, min ):
	global regionNameDictionary
	regionsIndexToProceed = []
	setupRegionSystem()
	if regionsToProceed.has_key('all') and regionsToProceed['all'] != 0.0:
		if len(regionNameDictionary) == 1:
			regionNameDictionary[1] = 'all'
		for key, value in sorted(regionNameDictionary.iteritems(), key=lambda (k,v): (v,k)):
			regionsIndexToProceed.append(1.0)
	else:
		for key, value in sorted(regionNameDictionary.iteritems(), key=lambda (k,v): (v,k)):
			if regionsToProceed.has_key(key):
				regionsIndexToProceed.append(regionsToProceed[key])
			elif key == 'empty':
				regionsIndexToProceed.append(0.0)
			else:
				regionsIndexToProceed.append(0.0)
	setscalarquantrandomuniformregion(quantName, regionsIndexToProceed, max, min)
	return

## Initialize vector quantity with uniform value in each region
#  Every regions that have been set but that are not used here will be considered as part of 'empty' region. Empty region will set the quantity to zero.
# @param quantName (string) name of the scalar quantity to set
# @param regionsToProceed (dictionary string(region name) => float (1 or 0)) 1 if region should be set up else 0.
def InitRandomUniformRegionVectorQuant(quantName, regionsToProceed ):
	global regionNameDictionary
	regionsIndexToProceed = []
	setupRegionSystem()
	if regionsToProceed.has_key('all') and regionsToProceed['all'] != 0.0:
		if len(regionNameDictionary) == 1:
			regionNameDictionary[1] = 'all'
		for key, value in sorted(regionNameDictionary.iteritems(), key=lambda (k,v): (v,k)):
			regionsIndexToProceed.append(1.0)
	else:
		for key, value in sorted(regionNameDictionary.iteritems(), key=lambda (k,v): (v,k)):
			if regionsToProceed.has_key(key):
				regionsIndexToProceed.append(regionsToProceed[key])
			elif key == 'empty':
				regionsIndexToProceed.append(0.0)
			else:
				regionsIndexToProceed.append(0.0)
	setvectorquantrandomuniformregion(quantName, regionsIndexToProceed)
	return




## Set the grid and initialise the region system with a (2D) regulqr Ngone given a number of sides, a radius, a rotation angle and a center.
#  The grid will fit exactly around the polygon.
# @param sides (int) the number of sides (e.g. 3 for triangle, 4 for square ...) (unitless)
# @param radius (float) the radius of the circumscribed circle of the Ngone (m)
# @param rotation (float) the rotation angle applied to the Ngone (center being the center of the circumscribed circle) (degree)
# @param Nx (int) the number of cells along the X-axis (unitless)
# @param Ny (int) the number of cells along the Y-axis (unitless)
# @param Nz (int) the number of cells along the Z-axis (unitless)
# @param Cz (float) the cell size along Z-axis (m)
# @param region (string) name of the region to assign to this Ngone (if zero, the region system will not be used)
def NgoneFit(sides, radius, rotation, Nx, Ny, Nz, Cz, region = 0):
	setgridsize(Nx, Ny, Nz)
	Xmin = 0
	Ymin = 0
	Xmax = 0
	Ymax = 0
	for n in range(0,sides):
		x = radius *math.cos(rotation*math.pi/180 + 2 * math.pi * n / sides)
		y = radius *math.sin(rotation*math.pi/180 + 2 * math.pi * n / sides)
		#print >> sys.stderr, "x: ", x, " y: ", y
		if Xmin > x :
			Xmin = x
		if Xmax < x :
			Xmax = x
		if Ymin > y :
			Ymin = y
		if Ymax < y :
			Ymax = y
	Cx = (Xmax - Xmin)/Nx
	Cy = (Ymax - Ymin)/Ny
	#print >> sys.stderr, "Xmin: ", Xmin, " Ymin: ", Ymin, "Xmax: ", Xmax, " Ymax: ", Ymax, " Cx: ", Cx, " Cy: ", Cy
	setcellsize(Cx, Cy, Cz)
	load('regions')
	#mask= [[[0 for x in range(Nz)] for x in range(Ny)] for x in range(Nx)]
	setupRegionSystem()
	global regionDefinition
	regionDefinition = []
	global regionNameDictionary
	regionNameDictionary[region] = float(len(regionNameDictionary))
	diagonal =  radius * math.cos(math.pi / sides)
	rotRad = rotation * math.pi / 180.
	for i in range(0,Nx):
		x = (i + 0.5)*Cx + Xmin
		regionDefinition.append([])
		for j in range(0,Ny):
			y = (j + 0.5)*Cy + Ymin
			theta = math.atan2(y, x)
			regionDefinition[i].append([])
			if theta < rotRad:
				theta += 2 *math.pi
			for n in range(0,sides):
				if theta <= 2*math.pi*(n+1)/sides + rotRad and theta > 2*math.pi*n/sides + rotRad:
					angle = -2*math.pi*(n+0.5)/sides - rotRad
					xx = x * math.cos(angle) - y * math.sin(angle)
					#print >> sys.stderr, i , j, x, y, xx, "<", radius * math.cos(math.pi / sides), angle/math.pi, theta/math.pi, regionNameDictionary[region]
					if xx < diagonal:
						for k in range(0,Nz):
							regionDefinition[i][j].append(regionNameDictionary[region])
					else:
						for k in range(0,Nz):
							regionDefinition[i][j].append(regionNameDictionary["empty"])
				
	setmask('regionDefinition', [regionDefinition])
	"""
	out = ""
	for j in range(0,Ny):
		for i in range(0,Nx):
			if regionDefinition[i][j][0] >= 2:
				out += "-"
			else:
				out += "."
		out += "\n"
	print >> sys.stderr, out
	print >> sys.stderr, "\n\n\n"
	
	regionDefinition = getarray('regionDefinition')
	out = ""
	for j in range(0,Ny):
		for i in range(0,Nx):
			if regionDefinition[0][i][j][0] >= 1:
				out += "-"
			else:
				out += "."
		out += "\n"
	print >> sys.stderr, out
	"""
	return 