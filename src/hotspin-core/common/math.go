//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

// This file implements some common mathematical functions.
// Author: Arne Vansteenkiste.

import (
	"math"
)

// Integer division rounded up
func DivUp(x, y int) int {
	return ((x - 1) / y) + 1
}

// True if not infinite and not NaN
func IsReal(f float32) bool {
	if math.IsInf(float64(f), 0) {
		return false
	}
	return !math.IsNaN(float64(f))
}

// True if not infinite, not NaN and not zero
func IsFinite(f float32) bool {
	if math.IsInf(float64(f), 0) {
		return false
	}
	if math.IsNaN(float64(f)) {
		return false
	}
	return f != 0
}

// True if f is (positive or negative) infinity
func IsInf(f float32) bool {
	return math.IsInf(float64(f), 0)
}

// Absolute value.
func Abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// Square root.
func Sqrt32(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

// Product of ints.
func Prod(a []int) int {
	p := 1
	for _, x := range a {
		p *= x
	}
	return p
}
