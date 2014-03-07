//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Author: Arne Vansteenkiste

import (
	"math"
)

// A 3-component vector
type vector [3]float64

func (v *vector) Set(x, y, z float64) {
	v[0] = x
	v[1] = y
	v[2] = z
}

func (v *vector) SetTo(other *vector) {
	v[0] = other[0]
	v[1] = other[1]
	v[2] = other[2]
}

func (a vector) Cross(b vector) vector {
	var cross vector
	cross[0] = a[1]*b[2] - a[2]*b[1]
	cross[1] = -a[0]*b[2] + a[2]*b[0]
	cross[2] = a[0]*b[1] - a[1]*b[0]
	return cross
}

func (a vector) Dot(b vector) float64 {
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

func (v vector) Norm() float64 {
	return float64(math.Sqrt(float64(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])))
}

func (v *vector) Normalize() {
	invnorm := 1. / v.Norm()
	v[0] *= invnorm
	v[1] *= invnorm
	v[2] *= invnorm
}

func (v *vector) Scale(r float64) {
	v[0] *= r
	v[1] *= r
	v[2] *= r
}

func (v *vector) Divide(r float64) {
	v[0] /= r
	v[1] /= r
	v[2] /= r
}

func (v *vector) Sub(other vector) {
	v[0] -= other[0]
	v[1] -= other[1]
	v[2] -= other[2]
}

func (v *vector) Add(other vector) {
	v[0] += other[0]
	v[1] += other[1]
	v[2] += other[2]
}
