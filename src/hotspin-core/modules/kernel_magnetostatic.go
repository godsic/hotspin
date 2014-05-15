//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Magnetostatic kernel
// Author: Arne Vansteenkiste

import (
	. "hotspin-core/common"
	"hotspin-core/host"
	"math"
	"time"
)

// Calculates the magnetostatic kernel by brute-force integration
// of magnetic charges over the faces and averages over cell volumes.
// Mesh should NOT yet be zero-padded.
func Kernel_Arne(size []int, cellsize []float64, pbc []int, accuracy_ int, kern *host.Array) {

	array := kern.Array
	accuracy := float64(accuracy_)

	// Shorthand
	Debug("calculating demag kernel")

	// Sanity check
	{
		Assert(size[0] >= 1 && size[1] >= 2 && size[2] >= 2)
		Assert(cellsize[X] > 0 && cellsize[Y] > 0 && cellsize[Z] > 0)
		Assert(pbc[X] >= 0 && pbc[Y] >= 0 && pbc[Z] >= 0)
		Assert(accuracy > 0)
		Assert(size[1]%2 == 0 && size[2]%2 == 0)
		if size[0] > 1 {
			Assert(size[0]%2 == 0)
		}
	}

	//	//----------------------------------
	//	log.Println(" ****   unit kernel **** ")
	//	array[X][X][0][0][0] = 1
	//	array[Y][Y][0][0][0] = 1
	//	array[Z][Z][0][0][0] = 1
	//	kernel[1][0] = kernel[0][1]
	//	kernel[2][0] = kernel[0][2]
	//	kernel[2][1] = kernel[1][2]
	//	return
	//	//------------------------------------

	// Field (destination) loop ranges
	var x1, x2, y1, y2, z1, z2 int
	// TODO: simplify
	{
		if pbc[X] == 0 {
			x1, x2 = -(size[X]-1)/2, (size[X]-1)/2
		} else {
			x1, x2 = -(size[X]*pbc[X] - 1), (size[X]*pbc[X] - 1)
		}
		if pbc[Y] == 0 {
			y1, y2 = -(size[Y]-1)/2, (size[Y]-1)/2
		} else {
			y1, y2 = -(size[Y]*pbc[Y] - 1), (size[Y]*pbc[Y] - 1)
		}
		if pbc[Z] == 0 {
			z1, z2 = -(size[Z]-1)/2, (size[Z]-1)/2
		} else {
			z1, z2 = -(size[Z]*pbc[Z] - 1), (size[Z]*pbc[Z] - 1)
		}
		// support for 2D simulations (thickness 1)
		if size[Z] == 1 && pbc[Z] == 0 {
			z2 = 0
		}
		Debug(" (ranges:", x1, x2, ",", y1, y2, ",", z1, z2, ")")
	}

	// smallest cell dimension is our typical length scale
	L := cellsize[X]
	{
		if cellsize[Y] < L {
			L = cellsize[Y]
		}
		if cellsize[Z] < L {
			L = cellsize[Z]
		}
	}

	// Start brute integration
	// 9 nested loops, does that stress you out?
	// Fortunately, the 5 inner ones usually loop over just one element.
	// It might be nice to get rid of that branching though.
	var (
		R, R2  [3]float64 // field and source cell center positions
		pole   [3]float64 // position of point charge on the surface
		points int        // counts used integration points
	)

	t0 := time.Now()

	for s := 0; s < 3; s++ { // source index Ksdxyz // TODO: make inner?
		Debug(".")
		u, v, w := s, (s+1)%3, (s+2)%3 // u = direction of source (s), v & w are the orthogonal directions

		for z := z1; z <= z2; z++ {
			zw := Wrap(z, size[Z])
			R[Z] = float64(z) * cellsize[Z]
			for y := y1; y <= y2; y++ {
				yw := Wrap(y, size[Y])
				R[Y] = float64(y) * cellsize[Y]

				for x := x1; x <= x2; x++ { // in each dimension, go from -(size-1)/2 to size/2 -1, wrapped.
					xw := Wrap(x, size[X])
					R[X] = float64(x) * cellsize[X]

					// choose number of integration points depending on how far we are from source.
					dx, dy, dz := delta(x)*cellsize[X], delta(y)*cellsize[Y], delta(z)*cellsize[Z]
					d := math.Sqrt(dx*dx + dy*dy + dz*dz)
					if d == 0 {
						d = L
					}
					maxSize := d / accuracy // maximum acceptable integration size
					nv := int(math.Max(cellsize[v]/maxSize, 1) + 0.5)
					nw := int(math.Max(cellsize[w]/maxSize, 1) + 0.5)
					nx := int(math.Max(cellsize[X]/maxSize, 1) + 0.5)
					ny := int(math.Max(cellsize[Y]/maxSize, 1) + 0.5)
					nz := int(math.Max(cellsize[Z]/maxSize, 1) + 0.5)
					// Stagger source and destination grids.
					// Massively improves accuracy, see note.
					nv *= 2
					nw *= 2

					Assert(nv > 0 && nw > 0 && nx > 0 && ny > 0 && nz > 0)

					scale := 1.0 / float64(nv*nw*nx*ny*nz)
					surface := cellsize[v] * cellsize[w] // the two directions perpendicular to direction s
					charge := surface * scale
					charge_4PI := charge / (4.0 * math.Pi)
					pu1 := cellsize[u] / 2. // positive pole center
					pu2 := -pu1             // negative pole center

					// Do surface integral over source cell, accumulate  in B
					var B [3]float64
					for i := 0; i < nv; i++ {
						pv := -(cellsize[v] / 2.) + cellsize[v]/float64(2*nv) + float64(i)*(cellsize[v]/float64(nv))
						pole[v] = pv
						for j := 0; j < nw; j++ {
							pw := -(cellsize[w] / 2.) + cellsize[w]/float64(2*nw) + float64(j)*(cellsize[w]/float64(nw))
							pole[w] = pw

							// Do volume integral over destination cell
							for α := 0; α < nx; α++ {
								rx := R[X] - cellsize[X]/2.0 + cellsize[X]/float64(2*nx) + (cellsize[X]/float64(nx))*float64(α)

								for β := 0; β < ny; β++ {
									ry := R[Y] - cellsize[Y]/2.0 + cellsize[Y]/float64(2*ny) + (cellsize[Y]/float64(ny))*float64(β)

									for γ := 0; γ < nz; γ++ {
										rz := R[Z] - cellsize[Z]/2.0 + cellsize[Z]/float64(2*nz) + (cellsize[Z]/float64(nz))*float64(γ)
										points++

										pole[u] = pu1
										R2[X], R2[Y], R2[Z] = rx-pole[X], ry-pole[Y], rz-pole[Z]
										r2 := R2[X]*R2[X] + R2[Y]*R2[Y] + R2[Z]*R2[Z]
										r := math.Sqrt(r2)
										qr := charge_4PI / (r2 * r)
										bx := R2[X] * qr
										by := R2[Y] * qr
										bz := R2[Z] * qr

										pole[u] = pu2
										R2[X], R2[Y], R2[Z] = rx-pole[X], ry-pole[Y], rz-pole[Z]
										r2 = R2[X]*R2[X] + R2[Y]*R2[Y] + R2[Z]*R2[Z]
										r = math.Sqrt(r2)
										qr = -charge_4PI / (r2 * r)
										B[X] += (bx + R2[X]*qr) // addition ordered for accuracy
										B[Y] += (by + R2[Y]*qr)
										B[Z] += (bz + R2[Z]*qr)

									}
								}
							}
						}
					}
					for d := 0; d < 3; d++ { // destination index Ksdxyz
						I := FullTensorIdx[s][d]
						array[I][xw][yw][zw] += B[d] // We have to ADD because there are multiple contributions in case of periodicity
					}
				}
			}
		}
	}
	t1 := time.Now()
	Debug("kernel used", points, "integration points")
	Debug("kernel calculation took", t1.Sub(t0))
	Debug("")
}

// closest distance between cells, given center distance d.
// if cells touch by just even a corner, the distance is zero.
func delta(d int) float64 {
	if d < 0 {
		d = -d
	}
	if d > 0 {
		d -= 1
	}
	return float64(d)
}

// Modulo-like function:
// Wraps an index to [0, max] by adding/subtracting a multiple of max.
func Wrap(number, max int) int {
	for number < 0 {
		number += max
	}
	for number >= max {
		number -= max
	}
	return number
}
