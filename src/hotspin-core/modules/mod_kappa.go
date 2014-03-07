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
	. "hotspin-core/common"
	. "hotspin-core/engine"
	"hotspin-core/gpu"
)

var inKappa = map[string]string{
	"T": "Teff",
}

var depsKappa = map[string]string{
	"Tc":      "Tc",
	"J":       "J",
	"n":       "n",
	"msat0":   "msat0",
	"msat0T0": "msat0T0",
}

var outKappa = map[string]string{
	"ϰ": "ϰ",
}

// Register this module
func init() {
	args := Arguments{inKappa, depsKappa, outKappa}
	RegisterModuleArgs("mfa/ϰ", "Temperature dependence of longitudinal susceptibility for finite J", args, LoadBrillouinKappaArgs)
}

func LoadBrillouinKappaArgs(e *Engine, args ...Arguments) {

	// make it automatic !!!
	var arg Arguments

	if len(args) == 0 {
		arg = Arguments{inKappa, depsKappa, outKappa}
	} else {
		arg = args[0]
	}
	//

	LoadFullMagnetization(e)
	LoadTemp(e, arg.Ins("T"))
	LoadKappa(e, arg.Outs("ϰ"))
	LoadMFAParams(e)

	e.Depends(arg.Outs("ϰ"), arg.Deps("Tc"), arg.Deps("msat0"), arg.Deps("J"), arg.Deps("msat0T0"), arg.Deps("n"), arg.Ins("T"))

	T := e.Quant(arg.Ins("T"))
	msat0 := e.Quant(arg.Deps("msat0"))
	msat0T0 := e.Quant(arg.Deps("msat0T0"))
	n := e.Quant(arg.Deps("n"))
	J := e.Quant(arg.Deps("J"))
	Tc := e.Quant(arg.Deps("Tc"))
	kappa := e.Quant(arg.Outs("ϰ"))

	kappa.SetUpdater(&kappaUpdater{kappa: kappa, msat0: msat0, msat0T0: msat0T0, T: T, Tc: Tc, J: J, n: n})

}

type kappaUpdater struct {
	kappa, msat0, msat0T0, T, Tc, J, n, gamma *Quant
}

func (u *kappaUpdater) Update() {
	kappa := u.kappa
	msat0 := u.msat0
	msat0T0 := u.msat0T0
	T := u.T
	Tc := u.Tc
	J := u.J
	n := u.n
	stream := kappa.Array().Stream
	kappa.Multiplier()[0] = msat0T0.Multiplier()[0] * msat0T0.Multiplier()[0] * Mu0 / (n.Multiplier()[0] * Kb)

	gpu.KappaAsync(kappa.Array(), msat0.Array(), msat0T0.Array(), T.Array(), Tc.Array(), J.Array(), n.Array(), msat0.Multiplier()[0], msat0T0.Multiplier()[0], Tc.Multiplier()[0], J.Multiplier()[0], stream)
	stream.Sync()
}
