//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Temperature dependance of equilibrium value of saturation magnetization for any finite J
// Author: Arne Vansteenkiste

import (
	. "hotspin-core/engine"
	"hotspin-core/gpu"
)

var inBrillouin = map[string]string{
	"T": "Teff",
}

var depsBrillouin = map[string]string{
	"Tc":      "Tc",
	"J":       "J",
	"msat0T0": "msat0T0",
}

var outBrillouin = map[string]string{
	"msat0": "msat0",
}

// Register this module
func init() {
	args := Arguments{inBrillouin, depsBrillouin, outBrillouin}
	RegisterModuleArgs("mfa/msat0", "Temperature dependence of equilibrium value of saturation magnetization for any finite J", args, LoadBrillouinArgs)
}

func LoadBrillouinArgs(e *Engine, args ...Arguments) {

	// make it automatic !!!
	var arg Arguments

	if len(args) == 0 {
		arg = Arguments{inBrillouin, depsBrillouin, outBrillouin}
	} else {
		arg = args[0]
	}
	//

	LoadFullMagnetization(e)
	LoadTemp(e, arg.Ins("T"))
	LoadMFAParams(e)

	T := e.Quant(arg.Ins("T"))
	J := e.Quant(arg.Deps("J"))
	Tc := e.Quant(arg.Deps("Tc"))
	msat0T0 := e.Quant(arg.Deps("msat0T0"))
	msat0 := e.Quant(arg.Outs("msat0"))

	e.Depends(arg.Outs("msat0"), arg.Deps("msat0T0"), arg.Deps("Tc"), arg.Deps("J"), arg.Ins("T"))

	msat0.SetUpdater(&BrillouinUpdater{msat0: msat0, msat0T0: msat0T0, T: T, Tc: Tc, J: J})

}

type BrillouinUpdater struct {
	msat0, msat0T0, T, Tc, J *Quant
}

func (u *BrillouinUpdater) Update() {
	msat0 := u.msat0
	msat0T0 := u.msat0T0
	T := u.T
	J := u.J
	Tc := u.Tc
	stream := msat0.Array().Stream
	gpu.BrillouinAsync(msat0.Array(), msat0T0.Array(), T.Array(), Tc.Array(), J.Array(), msat0.Multiplier()[0], msat0T0.Multiplier()[0], Tc.Multiplier()[0], J.Multiplier()[0], stream)
	stream.Sync()
}
