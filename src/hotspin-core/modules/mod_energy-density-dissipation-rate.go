//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

import (
	. "hotspin-core/common"
	. "hotspin-core/engine"
	"hotspin-core/gpu"
)

var inDF = map[string]string{
	"": "",
}

var depsDF = map[string]string{
	"R":       "R",
	"H_eff":   "H_eff",
	"msat0T0": "msat0T0",
}

var outDF = map[string]string{
	"Qmag": "Qmag",
}

// Register this module
func init() {
	args := Arguments{inDF, depsDF, outDF}
	RegisterModuleArgs("energy-density-dissipation-rate", "Energy density dissipation rate", args, LoadDFArgs)
}

// There is a problem, since LLB torque is normalized by msat0T0 (zero-temperature value), while LLG torque is normalized by msat
// This has to be explicitly accounted when module is loaded

func LoadDFArgs(e *Engine, args ...Arguments) {

	// make it automatic !!!
	var arg Arguments

	if len(args) == 0 {
		arg = Arguments{inDF, depsDF, outDF}
	} else {
		arg = args[0]
	}
	//

	// make sure the effective field is in place
	LoadHField(e)

	Qmagn := e.AddNewQuant(arg.Outs("Qmag"), SCALAR, FIELD, Unit("J/(s*m3)"), "The dissipative function")

	e.Depends(arg.Outs("Qmag"), arg.Deps("H_eff"), arg.Deps("msat0T0"), arg.Deps("R"))
	Qmagn.SetUpdater(&DFUpdater{
		Qmagn:   Qmagn,
		msat0T0: e.Quant(arg.Deps("msat0T0")),
		Heff:    e.Quant(arg.Deps("H_eff")),
		R:       e.Quant(arg.Deps("R"))})
}

type DFUpdater struct {
	Qmagn   *Quant
	msat0T0 *Quant
	Heff    *Quant
	R       *Quant
}

func (u *DFUpdater) Update() {

	// Account for msat0T0 multiplier, because it is a mask
	u.Qmagn.Multiplier()[0] = u.msat0T0.Multiplier()[0]
	// Account for - 2.0 * 0.5 * mu0
	u.Qmagn.Multiplier()[0] *= -Mu0
	// Account for multiplier in H_eff
	u.Qmagn.Multiplier()[0] *= u.Heff.Multiplier()[0]
	// Account for multiplier in R that should always be equal to the gyromagnetic ratio. Moreover the R is reduced to [1/s] units
	u.Qmagn.Multiplier()[0] *= u.R.Multiplier()[0]

	gpu.Dot(u.Qmagn.Array(),
		u.Heff.Array(),
		u.R.Array())

	// Finally. do Qmag = Qmag * msat0T0(r) to account spatial properties of msat0T0 that are hidden in the definition of the relaxation constants
	if !u.msat0T0.Array().IsNil() {
		gpu.Mul(u.Qmagn.Array(),
			u.Qmagn.Array(),
			u.msat0T0.Array())
	}
}
