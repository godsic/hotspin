//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// This file implements reduction operations on a quantity
// (average, min, max, ...)
// Author: Arne Vansteenkiste

import (
	. "hotspin-core/common"
	"hotspin-core/gpu"
)

// Superclass for all reduce updaters.
type ReduceUpdater struct {
	in, out *Quant
	reduce  gpu.Reductor
}

// New reducing updater.
// Automatically sets the dependency in -> out.
func NewReduceUpdater(in, out *Quant) *ReduceUpdater {
	checkKinds(in, FIELD, MASK)
	red := new(ReduceUpdater)
	red.in = in
	red.out = out
	red.reduce.Init(1, GetEngine().GridSize())
	GetEngine().Depends(out.Name(), in.Name())
	return red
}

// ________________________________________________________________________________ average

// Updates an average quantity
type AverageUpdater ReduceUpdater

// Returns an updater that writes the average of in to out
func NewAverageUpdater(in, out *Quant) Updater {
	return (*AverageUpdater)(NewReduceUpdater(in, out))
}

func (this *AverageUpdater) Update() {
	var sum float32 = 666
	if this.in.IsSpaceDependent() {
		if this.in.nComp == 1 {
			sum = this.reduce.Sum(this.in.Array())
			this.out.SetScalar(float64(sum) * this.in.multiplier[0] / float64(GetEngine().NCell()))
		} else {
			for c := 0; c < this.in.nComp; c++ {
				sum := this.reduce.Sum(&(this.in.Array().Comp[c]))
				this.out.SetComponent(c, float64(sum)*this.in.multiplier[c]/float64(GetEngine().NCell()))
			}
		}
	} else {
		// The mask is not space dependant, so just copy the multiplier from the 'in' into the 'out'
		if this.in.nComp == 1 {
			this.out.SetScalar(this.in.multiplier[0])
		} else {
			for c := 0; c < this.in.nComp; c++ {
				this.out.SetComponent(c, this.in.multiplier[c])
			}
		}
	}
}

// ________________________________________________________________________________ sum

// Updates an average quantity
type SumReduceUpdater ReduceUpdater

// Returns an updater that writes the average of in to out
func NewSumReduceUpdater(in, out *Quant) Updater {
	return (*SumReduceUpdater)(NewReduceUpdater(in, out))
}

func (this *SumReduceUpdater) Update() {
	var sum float32 = 666
	if this.in.IsSpaceDependent() {
		if this.in.nComp == 1 {
			sum = this.reduce.Sum(this.in.Array())
			this.out.SetScalar(float64(sum) * this.in.multiplier[0])
		} else {
			for c := 0; c < this.in.nComp; c++ {
				sum := this.reduce.Sum(&(this.in.Array().Comp[c]))
				this.out.SetComponent(c, float64(sum)*this.in.multiplier[c])
			}
		}
	} else {
		// The mask is not space dependant, so just copy the multiplier from the 'in' into the 'out'
		if this.in.nComp == 1 {
			this.out.SetScalar(this.in.multiplier[0])
		} else {
			for c := 0; c < this.in.nComp; c++ {
				this.out.SetComponent(c, this.in.multiplier[c])
			}
		}
	}
}

// ________________________________________________________________________________ maxabs

// Updates a maximum of absolute values
type MaxAbsUpdater ReduceUpdater

// Returns an updater that writes the maximum of absolute values of in to out
func NewMaxAbsUpdater(in, out *Quant) Updater {
	return (*MaxAbsUpdater)(NewReduceUpdater(in, out))
}

func (this *MaxAbsUpdater) Update() {
	var max float64
	for c := 0; c < this.in.nComp; c++ {
		compMax := float64(this.reduce.MaxAbs(this.in.Array().Component(c))) * this.in.multiplier[c]
		if compMax > max {
			max = compMax
		}
	}
	this.out.SetScalar(float64(max))
}

// ________________________________________________________________________________ maxnorm

// Updates the maximum norm of a 3-vector array
type MaxNormUpdater ReduceUpdater

// Returns an updater that writes the maximum of absolute values of in to out
func NewMaxNormUpdater(in, out *Quant) Updater {
	Assert(in.NComp() == 3)
	return (*MaxNormUpdater)(NewReduceUpdater(in, out))
}

func (this *MaxNormUpdater) Update() {
	Assert(this.in.multiplier[0] == this.in.multiplier[1] && this.in.multiplier[1] == this.in.multiplier[2])
	arr := this.in.Array()
	max := float64(this.reduce.MaxNorm(arr)) * this.in.multiplier[0]
	this.out.SetScalar(float64(max))
}
