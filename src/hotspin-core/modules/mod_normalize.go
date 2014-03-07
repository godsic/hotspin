//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// temperature dependence of longitudinal susceptibility as follows from mean-field approximation
// Author: Mykola Dvornik

import (
	. "hotspin-core/engine"
	"hotspin-core/gpu"
)

var inNorm = map[string]string{
	"in": "mf",
}

var depsNorm = map[string]string{}

var outNorm = map[string]string{}

// Register this module
func init() {
	args := Arguments{inNorm, depsNorm, outNorm}
	RegisterModuleArgs("normalizer", "Quantity normalizer", args, LoadNorm)
}

func LoadNorm(e *Engine, args ...Arguments) {

	// make it automatic !!!
	var arg Arguments

	if len(args) == 0 {
		arg = Arguments{inNorm, depsNorm, outNorm}
	} else {
		arg = args[0]
	}

	inout := e.Quant(arg.Ins("in"))

	inout.SetUpdater(&NormUpdater{inout: inout})

}

type NormUpdater struct {
	inout *Quant
}

func (u *NormUpdater) Update() {
	gpu.Normalize(u.inout.Array())
}
