//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Ms -> Ts mapping using Mean Field Approximation
// Author: Mykola Dvornik

import (
	. "hotspin-core/engine"
	"hotspin-core/gpu"
)

var inTs = map[string]string{
	"msat": "msat",
}

var depsTs = map[string]string{
	"Tc":      "Tc",
	"J":       "J",
	"msat0T0": "msat0T0",
}

var outTs = map[string]string{
	"Ts": "Ts",
}

// Register this module
func init() {
	args := Arguments{inTs, depsTs, outTs}
	RegisterModuleArgs("mfa/Ts", "Ms -> Ts mapping using Mean Field Approximation", args, LoadTsArgs)
}

func LoadTsArgs(e *Engine, args ...Arguments) {

	// make it automatic !!!
	var arg Arguments

	if len(args) == 0 {
		arg = Arguments{inTs, depsTs, outTs}
	} else {
		arg = args[0]
	}
	//

	LoadFullMagnetization(e)
	LoadTemp(e, arg.Outs("Ts"))
	LoadMFAParams(e)

	msat := e.Quant(arg.Ins("msat"))
	J := e.Quant(arg.Deps("J"))
	Tc := e.Quant(arg.Deps("Tc"))
	msat0T0 := e.Quant(arg.Deps("msat0T0"))
	Ts := e.Quant(arg.Outs("Ts"))

	e.Depends(arg.Outs("Ts"), arg.Deps("msat0T0"), arg.Deps("Tc"), arg.Deps("J"), arg.Ins("msat"))

	Ts.SetUpdater(&TsUpdater{Ts: Ts, msat: msat, msat0T0: msat0T0, Tc: Tc, J: J})

}

type TsUpdater struct {
	Ts, msat, msat0T0, Tc, J *Quant
}

func (u *TsUpdater) Update() {
	msat := u.msat
	msat0T0 := u.msat0T0
	Ts := u.Ts
	J := u.J
	Tc := u.Tc
	gpu.TsSync(Ts.Array(), msat.Array(), msat0T0.Array(), Tc.Array(), J.Array(), msat.Multiplier()[0], msat0T0.Multiplier()[0], Tc.Multiplier()[0], J.Multiplier()[0])
}
