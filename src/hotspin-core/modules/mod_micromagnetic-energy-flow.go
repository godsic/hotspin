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

var inMMEF = map[string]string{
	"": "",
}

var depsMMEF = map[string]string{
	"R":       "R",
	"msat0T0": "msat0T0",
	"m":      "m",
}

var outMMEF = map[string]string{
	"q_mm": "q_mm",
}

// Register this module
func init() {
	args := Arguments{inMMEF, depsMMEF, outMMEF}
	RegisterModuleArgs("micromagnetic-energy-flow", "Micromagnetic energy density dissipation rate", args, LoadMMEFArgs)
}

// There is a problem, since LLB torque is normalized by msat0T0 (zero-temperature value), while LLG torque is normalized by msat
// This has to be explicitly accounted when module is loaded

func LoadMMEFArgs(e *Engine, args ...Arguments) {

	// make it automatic !!!
	var arg Arguments

	if len(args) == 0 {
		arg = Arguments{inMMEF, depsMMEF, outMMEF}
	} else {
		arg = args[0]
	}

	Qmm := e.AddNewQuant(arg.Outs("q_mm"), SCALAR, FIELD, Unit("J/(s*m3)"), "Micromagnetic energy density dissipation rate")

	e.Depends(arg.Outs("q_mm"), arg.Deps("m"), arg.Deps("msat0T0"), arg.Deps("R"))
	Qmm.SetUpdater(&MMEFUpdater{
		Qmm:     Qmm,
		m:      e.Quant(arg.Deps("m")),
		msat0T0: e.Quant(arg.Deps("msat0T0")),
		R:       e.Quant(arg.Deps("R"))})
}

type MMEFUpdater struct {
	Qmm     *Quant
	m      *Quant
	msat0T0 *Quant
	R       *Quant
}

func (u *MMEFUpdater) Update() {

	// Account for msat0T0 multiplier in both relaxation tensor and magnetization vector
	u.Qmm.Multiplier()[0] = u.msat0T0.Multiplier()[0]
	u.Qmm.Multiplier()[0] *= u.msat0T0.Multiplier()[0]

	// Account for mu0
	u.Qmm.Multiplier()[0] *= Mu0
	// Account for multiplier in R that should always be equal to the gyromagnetic ratio. Moreover the R is reduced to [1/s] units
	u.Qmm.Multiplier()[0] *= u.R.Multiplier()[0]

	gpu.Dot(u.Qmm.Array(),
		u.m.Array(),
		u.R.Array())

	// Finally, do Qmm = Qmm * msat0T0(r)^2 to account spatial properties of msat0T0 that are hidden in the definitions of the relaxation constants and magnertization vector
	if !u.msat0T0.Array().IsNil() {
		gpu.Mul(u.Qmm.Array(),
			u.Qmm.Array(),
			u.msat0T0.Array())
		gpu.Mul(u.Qmm.Array(),
			u.Qmm.Array(),
			u.msat0T0.Array())
	}
}
