//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Module implements nonconservative zero-order local damping
// Authors: Mykola Dvornik, Arne Vansteenkiste

import (
	. "hotspin-core/engine"
	"hotspin-core/gpu"
)

var inLocal00NC = map[string]string{
	"α": "α",
}

var depsLocal00NC = map[string]string{
	"γ_LL":    "γ_LL",
	"msat0T0": "msat0T0",
	"H_eff":   "H_eff",
}

var outLocal00NC = map[string]string{
	"llbar_local00nc": "llbar_local00nc",
}

// Register this module

func init() {
	args := Arguments{inLocal00NC, depsLocal00NC, outLocal00NC}
	RegisterModuleArgs("llbar/damping/nonconservative/00/local", "LLBar nonconservative zero-order local relaxation term", args, LoadLLBarLocal00NCArgs)
}

func LoadLLBarLocal00NCArgs(e *Engine, args ...Arguments) {

	// make it automatic !!!
	var arg Arguments

	if len(args) == 0 {
		arg = Arguments{inLocal00NC, depsLocal00NC, outLocal00NC}
	} else {
		arg = args[0]
	}
	//

	LoadHField(e)
	LoadFullMagnetization(e)
	LoadGammaLL(e)

	// ============ Quantities =============
	if !e.HasQuant(arg.Ins("α")) {
		e.AddNewQuant(arg.Ins("α"), VECTOR, MASK, Unit(""), "LLBar zero-order local relaxation diagonal tensor")
	}
	lambda := e.Quant(arg.Ins("α"))
	msat0T0 := e.Quant(arg.Deps("msat0T0"))
	H := e.Quant(arg.Deps("H_eff"))
	gammaLL := e.Quant(arg.Deps("γ_LL"))
	llbar_local00nc := e.AddNewQuant(arg.Outs("llbar_local00nc"), VECTOR, FIELD, Unit("/s"), "Landau-Lifshitz-Baryakhtar nonconservative zero-order local relaxation term")

	// =============== Dependencies =============
	e.Depends(arg.Outs("llbar_local00nc"), arg.Deps("H_eff"), arg.Deps("γ_LL"), arg.Deps("msat0T0"), arg.Ins("α"))

	// ============ Updating the torque =============
	upd := &LLBarLocal00NCUpdater{llbar_local00nc: llbar_local00nc, H: H, gammaLL: gammaLL, msat0T0: msat0T0, lambda: lambda}
	llbar_local00nc.SetUpdater(upd)
}

type LLBarLocal00NCUpdater struct {
	llbar_local00nc, H, gammaLL, msat0T0, lambda *Quant
}

func (u *LLBarLocal00NCUpdater) Update() {

	llbar_local00nc := u.llbar_local00nc
	gammaLL := u.gammaLL.Scalar()
	heff := u.H

	// put gamma in multiplier to avoid additional multiplications
	multiplierBT := llbar_local00nc.Multiplier()
	for i := range multiplierBT {
		multiplierBT[i] = gammaLL
	}

	lambda := u.lambda
	msat0T0 := u.msat0T0

	gpu.LLBarLocal00NC(llbar_local00nc.Array(),
		heff.Array(),
		msat0T0.Array(),
		lambda.Array(),
		lambda.Multiplier())

	llbar_local00nc.Array().Sync()
}
