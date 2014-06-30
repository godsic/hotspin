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
	"fmt"
	. "hotspin-core/common"
	"hotspin-core/host"
	"math"
	"sync/atomic"
	"time"
)

// Calculates the magnetostatic kernel by brute-force integration
// of magnetic charges over the faces and averages over cell volumes.
// Mesh should NOT yet be zero-padded.
func Kernel_Arne(size []int, cellsize []float64, pbc []int, accuracy_ int, kernel *host.Array) {

	sanityCheck(cellsize, pbc)

	array := kernel.Array
	accuracy := float64(accuracy_)

	// Shorthand
	Debug("calculating demag kernel")

	// Sanity check
	{
		Assert(size[Z] > 0 && size[Y] > 0 && size[X] > 0)
		Assert(cellsize[X] > 0 && cellsize[Y] > 0 && cellsize[Z] > 0)
		Assert(pbc[X] >= 0 && pbc[Y] >= 0 && pbc[Z] >= 0)
		Assert(accuracy > 0)
	}

	// Field (destination) loop ranges
	r1, r2 := kernelRanges(size, pbc)

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

	progress, progmax := 0, (1+(r2[Y]-r1[Y]))*(1+(r2[Z]-r1[Z])) // progress bar
	done := make(chan struct{}, 3)                              // parallel calculation of one component done?

	points := int64(0)

	t0 := time.Now()

	// Start brute integration
	// 9 nested loops, does that stress you out?
	// Fortunately, the 5 inner ones usually loop over just one element.

	for s := 0; s < 3; s++ { // source index Ksdxyz (parallelized over)
		go func(s int) {
			u, v, w := s, (s+1)%3, (s+2)%3 // u = direction of source (s), v & w are the orthogonal directions
			var (
				R, R2 [3]float64 // field and source cell center positions
				pole  [3]float64 // position of point charge on the surface
			)

			for z := r1[Z]; z <= r2[Z]; z++ {
				zw := wrap(z, size[Z])
				// skip one half, reconstruct from symmetry later
				// check on wrapped index instead of loop range so it also works for PBC
				if zw > size[Z]/2 {
					if s == 0 {
						progress += (1 + (r2[Y] - r1[Y]))
					}
					continue
				}
				R[Z] = float64(z) * cellsize[Z]

				for y := r1[Y]; y <= r2[Y]; y++ {

					if s == 0 { // show progress of only one component
						progress++
						Progress(progress, progmax, "calculating demag kernel")
					}

					yw := wrap(y, size[Y])
					if yw > size[Y]/2 {
						continue
					}
					R[Y] = float64(y) * cellsize[Y]

					for x := r1[X]; x <= r2[X]; x++ {
						xw := wrap(x, size[X])
						if xw > size[X]/2 {
							continue
						}
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

						scale := 1 / float64(nv*nw*nx*ny*nz)
						surface := cellsize[v] * cellsize[w] // the two directions perpendicular to direction s
						charge := surface * scale
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
									rx := R[X] - cellsize[X]/2 + cellsize[X]/float64(2*nx) + (cellsize[X]/float64(nx))*float64(α)

									for β := 0; β < ny; β++ {
										ry := R[Y] - cellsize[Y]/2 + cellsize[Y]/float64(2*ny) + (cellsize[Y]/float64(ny))*float64(β)

										for γ := 0; γ < nz; γ++ {
											rz := R[Z] - cellsize[Z]/2 + cellsize[Z]/float64(2*nz) + (cellsize[Z]/float64(nz))*float64(γ)
											atomic.AddInt64(&points, 1)

											pole[u] = pu1
											R2[X], R2[Y], R2[Z] = rx-pole[X], ry-pole[Y], rz-pole[Z]
											r := math.Sqrt(R2[X]*R2[X] + R2[Y]*R2[Y] + R2[Z]*R2[Z])
											qr := charge / (4 * math.Pi * r * r * r)
											bx := R2[X] * qr
											by := R2[Y] * qr
											bz := R2[Z] * qr

											pole[u] = pu2
											R2[X], R2[Y], R2[Z] = rx-pole[X], ry-pole[Y], rz-pole[Z]
											r = math.Sqrt(R2[X]*R2[X] + R2[Y]*R2[Y] + R2[Z]*R2[Z])
											qr = -charge / (4 * math.Pi * r * r * r)
											B[X] += (bx + R2[X]*qr) // addition ordered for accuracy
											B[Y] += (by + R2[Y]*qr)
											B[Z] += (bz + R2[Z]*qr)

										}
									}
								}
							}
						}
						for d := s; d < 3; d++ { // destination index Ksdxyz
							I := FullTensorIdx[s][d]
							array[I][xw][yw][zw] += B[d] // += needed in case of PBC
						}
					}
				}
			}
			done <- struct{}{} // notify parallel computation of this component is done
		}(s)
	}
	// wait for all 3 components to finish
	<-done
	<-done
	<-done

	// Reconstruct skipped parts from symmetry (X)
	for z := 0; z < size[Z]; z++ {
		for y := 0; y < size[Y]; y++ {
			for x := size[X]/2 + 1; x < size[X]; x++ {
				x2 := size[X] - x
				array[XX][x][y][z] = +array[XX][x2][y][z]
				array[XY][x][y][z] = -array[XY][x2][y][z]
				array[XZ][x][y][z] = -array[XZ][x2][y][z]
				array[YY][x][y][z] = +array[YY][x2][y][z]
				array[YZ][x][y][z] = +array[YZ][x2][y][z]
				array[ZZ][x][y][z] = +array[ZZ][x2][y][z]
			}
		}
	}

	// Reconstruct skipped parts from symmetry (Y)
	for z := 0; z < size[Z]; z++ {
		for y := size[Y]/2 + 1; y < size[Y]; y++ {
			y2 := size[Y] - y
			for x := 0; x < size[X]; x++ {
				array[XX][x][y][z] = +array[XX][x][y2][z]
				array[XY][x][y][z] = -array[XY][x][y2][z]
				array[XZ][x][y][z] = +array[XZ][x][y2][z]
				array[YY][x][y][z] = +array[YY][x][y2][z]
				array[YZ][x][y][z] = -array[YZ][x][y2][z]
				array[ZZ][x][y][z] = +array[ZZ][x][y2][z]

			}
		}
	}

	// Reconstruct skipped parts from symmetry (Z)
	for z := size[Z]/2 + 1; z < size[Z]; z++ {
		z2 := size[Z] - z
		for y := 0; y < size[Y]; y++ {
			for x := 0; x < size[X]; x++ {
				array[XX][x][y][z] = +array[XX][x][y][z2]
				array[XY][x][y][z] = +array[XY][x][y][z2]
				array[XZ][x][y][z] = -array[XZ][x][y][z2]
				array[YY][x][y][z] = +array[YY][x][y][z2]
				array[YZ][x][y][z] = -array[YZ][x][y][z2]
				array[ZZ][x][y][z] = +array[ZZ][x][y][z2]
			}
		}
	}

	t1 := time.Now()

	Debug("kernel used", points, "integration points")
	Debug("kernel calculation took", t1.Sub(t0))
	Debug("")
}

// integration ranges for kernel. size=kernelsize, so padded for no PBC, not padded for PBC
func kernelRanges(size, pbc []int) (r1, r2 [3]int) {
	for c := 0; c < 3; c++ {
		if pbc[c] == 0 {
			r1[c], r2[c] = -(size[c]-1)/2, (size[c]-1)/2
		} else {
			r1[c], r2[c] = -(size[c]*pbc[c] - 1), (size[c]*pbc[c] - 1) // no /2 here, or we would take half right and half left image
		}
	}
	// support for 2D simulations (thickness 1)
	if size[Z] == 1 && pbc[Z] == 0 {
		r2[Z] = 0
	}
	return
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

// wraps an index to [0, max] by adding/subtracting a multiple of max.
func wrap(number, max int) int {
	for number < 0 {
		number += max
	}
	for number >= max {
		number -= max
	}
	return number
}

const maxAspect = 100.0 // maximum sane cell aspect ratio

func sanityCheck(cellsize []float64, pbc []int) {
	a3 := cellsize[X] / cellsize[Y]
	a2 := cellsize[Y] / cellsize[Z]
	a1 := cellsize[Z] / cellsize[X]

	aMax := math.Max(a1, math.Max(a2, a3))
	aMin := math.Min(a1, math.Min(a2, a3))

	if aMax > maxAspect || aMin < 1./maxAspect {
		panic(InputErr(fmt.Sprint("Unrealistic cell aspect ratio:", cellsize)))
	}
}
