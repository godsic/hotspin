//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// The effective field responsible for exchange longitudinal relaxation
// Author: Mykola Dvornik

import (
	. "hotspin-core/engine"
	"hotspin-core/gpu"
)

var inLongField = map[string]string{
	"T": "Teff",
}

var depsLongField = map[string]string{
	"Tc":      "Tc",
	"m":      "m",
	"J":       "J",
	"n":       "n",
	"msat0T0": "msat0T0",
}

var outLongField = map[string]string{
	"H_lf": "H_lf",
}

// Register this module
func init() {
	args := Arguments{inLongField, depsLongField, outLongField}
	RegisterModuleArgs("mfa/longfield", "The effective field responsible for exchange longitudinal relaxation", args, LoadLongFieldArgs)
}

func LoadLongFieldArgs(e *Engine, args ...Arguments) {

	// make it automatic !!!
	var arg Arguments

	if len(args) == 0 {
		arg = Arguments{inBrillouin, depsBrillouin, outBrillouin}
	} else {
		arg = args[0]
	}
	//

	LoadHField(e)
	LoadFullMagnetization(e)
	LoadTemp(e, arg.Ins("T"))
	LoadMFAParams(e)

	T := e.Quant(arg.Ins("T"))
	Tc := e.Quant(arg.Deps("Tc"))
	m := e.Quant(arg.Deps("m"))
	J := e.Quant(arg.Deps("J"))
	n := e.Quant(arg.Deps("n"))
	msat0T0 := e.Quant(arg.Deps("msat0T0"))
	Hlf := e.AddNewQuant(arg.Outs("H_lf"), VECTOR, FIELD, Unit("A/m"), "longitudinal exchange field")
	e.Depends(arg.Outs("H_lf"), arg.Deps("J"), arg.Deps("n"), arg.Deps("m"), arg.Deps("Tc"), arg.Deps("msat0T0"), arg.Ins("T"))

	Hlf.SetUpdater(&LongFieldUpdater{m: m, J: J, Hlf: Hlf, msat0T0: msat0T0, n: n, Tc: Tc, T: T})

}

type LongFieldUpdater struct {
	m, J, Hlf, msat0T0, n, Tc, T *Quant
}

func (u *LongFieldUpdater) Update() {
	m := u.m
	J := u.J
	n := u.n
	Hlf := u.Hlf
	msat0T0 := u.msat0T0
	Tc := u.Tc
	T := u.T
	stream := u.Hlf.Array().Stream

	gpu.LongFieldAsync(Hlf.Array(), m.Array(), msat0T0.Array(), J.Array(), n.Array(), Tc.Array(), T.Array(), msat0T0.Multiplier()[0], J.Multiplier()[0], n.Multiplier()[0], Tc.Multiplier()[0], T.Multiplier()[0], stream)
	stream.Sync()
}
