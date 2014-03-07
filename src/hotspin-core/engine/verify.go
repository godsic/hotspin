//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// This file provides a number of func(*Quant)'s that can be used as input verifiers.

// Author: Arne Vansteenkiste

import (
	. "hotspin-core/common"
)

// Panics if a multiplier is zero
func NonZero(q *Quant) {
	nonzero := false
	for _, v := range q.multiplier {
		if v != 0 {
			nonzero = true
			break
		}
	}
	if !nonzero {
		panic(InputErr(q.Name() + " should be non-zero"))
	}
}

// Panics if a multiplier is <= 0
func Positive(q *Quant) {
	for _, v := range q.multiplier {
		if v <= 0 {
			panic(InputErr(q.Name() + " should be positive"))
		}
	}
}

// Panics if a multiplier is not a positive or zero integer
func Uint(q *Quant) {
	for _, v := range q.multiplier {
		if v < 0 || float64(int(v)) != v {
			panic(InputErr(q.Name() + " should be integer >= 0"))
		}
	}
}

// Panics if a multiplier is not a strictly positive integer
func PosInt(q *Quant) {
	for _, v := range q.multiplier {
		if v <= 0 || float64(int(v)) != v {
			panic(InputErr(q.Name() + " should be integer > 0"))
		}
	}
}

// Panics if a multiplier is not an integer
func Int(q *Quant) {
	for _, v := range q.multiplier {
		if float64(int(v)) != v {
			panic(InputErr(q.Name() + " should be integer"))
		}
	}
}

// Panics if a multiplier is < 0
func NonNegative(q *Quant) {
	for _, v := range q.multiplier {
		if v < 0 {
			panic(InputErr(q.Name() + " should be non-negative"))
		}
	}
}

func AtLeast1(q *Quant) {
	for _, v := range q.multiplier {
		if v < 1 {
			panic(InputErr(q.Name() + " should be >= 1"))
		}
	}
}
