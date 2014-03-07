//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// This file implements the dot product of quantities
// Author: Arne Vansteenkiste

import (
	"hotspin-core/gpu"
)

type SDotUpdater struct {
	sum              *Quant
	parent1, parent2 *Quant
	scaling          float64
	reduce           gpu.Reductor
}

// takes dot product
func NewSDotUpdater(sum, parent1, parent2 *Quant, scaling float64) Updater {
	u := new(SDotUpdater)
	u.sum = sum
	u.parent1 = parent1
	u.parent2 = parent2
	u.scaling = scaling
	u.reduce.Init(1, GetEngine().GridSize())
	GetEngine().Depends(sum.Name(), parent1.Name(), parent2.Name())
	return u
}

func (u *SDotUpdater) Update() {
	parent1 := u.parent1
	parent2 := u.parent2

	if !parent1.IsSpaceDependent() {
		parent1, parent2 = parent2, parent1
	}

	u.sum.multiplier[0] = 0
	if parent2.IsSpaceDependent() {
		for c := 0; c < u.parent1.NComp(); c++ {
			par1Mul := u.parent1.multiplier[c]
			par2Mul := u.parent2.multiplier[c]
			par1Comp := u.parent1.array.Component(c)
			par2Comp := u.parent2.array.Component(c)
			u.sum.multiplier[0] += float64(u.reduce.Dot(par1Comp, par2Comp)) * par1Mul * par2Mul * u.scaling
		}
	} else {
		for c := 0; c < u.parent1.NComp(); c++ {
			par1Mul := u.parent1.multiplier[c]
			par2Mul := u.parent2.multiplier[c]
			par1Comp := u.parent1.array.Component(c)
			//fmt.Println("u.sum.multiplier[0] += ", float64(u.reduce.Sum(par1Comp)) , par1Mul , par2Mul , u.scaling)
			u.sum.multiplier[0] += float64(u.reduce.Sum(par1Comp)) * par1Mul * par2Mul * u.scaling
		}
	}
}
