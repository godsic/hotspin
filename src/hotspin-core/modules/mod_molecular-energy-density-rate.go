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

var inEF = map[string]string{
	"": "",
}

var depsEF = map[string]string{
	"R":  "R",
	"m":  "m",
	"Tc": "Tc",
	"J":  "J",
	"n":  "n",
}

var outEF = map[string]string{
	"w": "w",
}

// Register this module
func init() {
	args := Arguments{inEF, depsEF, outEF}
	RegisterModuleArgs("mfa/molecular-energy-density-rate", "Molecular energy density rate", args, LoadEFArgs)
}

// There is a problem, since LLB torque is normalized by msat0T0 (zero-temperature value), while LLG torque is normalized by msat
// This has to be explicitly accounted when module is loaded

func LoadEFArgs(e *Engine, args ...Arguments) {

	// make it automatic !!!
	var arg Arguments

	if len(args) == 0 {
		arg = Arguments{inEF, depsEF, outEF}
	} else {
		arg = args[0]
	}
	//

	// make sure the effective field is in place
	LoadMFAParams(e)

	q_s := e.AddNewQuant(arg.Outs("w"), SCALAR, FIELD, Unit("J/(s*m3)"), "Spins energy density dissipation rate according to MFA")

	e.Depends(arg.Outs("w"), arg.Deps("m"), arg.Deps("R"), arg.Deps("J"), arg.Deps("Tc"), arg.Deps("n"))
	q_s.SetUpdater(&EFUpdater{
		q_s: q_s,
		J:   e.Quant(arg.Deps("J")),
		Tc:  e.Quant(arg.Deps("Tc")),
		n:   e.Quant(arg.Deps("n")),
		R:   e.Quant(arg.Deps("R")),
		m:   e.Quant(arg.Deps("m"))})
}

type EFUpdater struct {
	q_s *Quant
	J   *Quant
	Tc  *Quant
	n   *Quant
	R   *Quant
	m   *Quant
}

func (u *EFUpdater) Update() {

	Tc := u.Tc.Multiplier()[0]
	n := u.n.Multiplier()[0]

	// Spin should be accounted in the kernel since it enters S(S+1) term
	mult := 3.0 * Kb * Tc * n

	// Account for the dissipation term multiplier, normally = gamma_LL
	u.q_s.Multiplier()[0] = u.R.Multiplier()[0]
	u.q_s.Multiplier()[0] *= u.m.Multiplier()[0]
	u.q_s.Multiplier()[0] *= mult

	stream := u.q_s.Array().Stream

	gpu.EnergyFlowAsync(u.q_s.Array(),
		u.m.Array(),
		u.R.Array(),
		u.Tc.Array(),
		u.J.Array(),
		u.n.Array(),
		u.J.Multiplier()[0],
		stream)
	stream.Sync()
}
