//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Module implements nonconservative second-order local damping
// Authors: Mykola Dvornik

import (
	. "hotspin-core/engine"
	"hotspin-core/gpu"
)

var inLocal02NC = map[string]string{
	"μ∥": "μ∥",
}

var depsLocal02NC = map[string]string{
	"γ_LL":    "γ_LL",
	"msat0T0": "msat0T0",
	"mf":      "mf",
	"H_eff":   "H_eff",
}

var outLocal02NC = map[string]string{
	"llbar_local02nc": "llbar_local02nc",
}

// Register this module
func init() {
	args := Arguments{inLocal02NC, depsLocal02NC, outLocal02NC}
	RegisterModuleArgs("llbar/damping/nonconservative/02/local", "LLBar nonconservative second-order local relaxation term", args, LoadLLBarLocal02NCArgs)
}

func LoadLLBarLocal02NCArgs(e *Engine, args ...Arguments) {

	// make it automatic !!!
	var arg Arguments

	if len(args) == 0 {
		arg = Arguments{inLocal02NC, depsLocal02NC, outLocal02NC}
	} else {
		arg = args[0]
	}
	//

	LoadHField(e)
	LoadFullMagnetization(e)
	LoadGammaLL(e)

	// =========== Quantities =============
	if !e.HasQuant(arg.Ins("μ∥")) {
		e.AddNewQuant(arg.Ins("μ∥"), VECTOR, MASK, Unit(""), "LLBar second-order local nonconservative relaxation diagonal tensor")
	}
	mu := e.Quant(arg.Ins("μ∥"))
	msat0T0 := e.Quant(arg.Deps("msat0T0"))
	mf := e.Quant(arg.Deps("mf"))
	H := e.Quant(arg.Deps("H_eff"))
	gammaLL := e.Quant(arg.Deps("γ_LL"))
	llbar_local02nc := e.AddNewQuant(arg.Outs("llbar_local02nc"), VECTOR, FIELD, Unit("/s"), "Landau-Lifshitz-Baryakhtar nonconservative second-order local relaxation term")

	// ============ Dependencies =============
	e.Depends(arg.Outs("llbar_local02nc"), arg.Deps("mf"), arg.Deps("H_eff"), arg.Deps("γ_LL"), arg.Deps("msat0T0"), arg.Ins("μ∥"))
	// ============ Updating the torque =============
	upd := &LLBarLocal02NCUpdater{llbar_local02nc: llbar_local02nc, mf: mf, H: H, gammaLL: gammaLL, msat0T0: msat0T0, mu: mu}
	llbar_local02nc.SetUpdater(upd)
}

type LLBarLocal02NCUpdater struct {
	llbar_local02nc, mf, H, gammaLL, msat0T0, mu *Quant
}

func (u *LLBarLocal02NCUpdater) Update() {

	llbar_local02nc := u.llbar_local02nc
	gammaLL := u.gammaLL.Scalar()
	m := u.mf
	heff := u.H

	// put gamma in multiplier to avoid additional multiplications
	multiplierBT := llbar_local02nc.Multiplier()
	for i := range multiplierBT {
		multiplierBT[i] = gammaLL
	}

	mu := u.mu
	msat0T0 := u.msat0T0

	gpu.LLBarLocal02NC(llbar_local02nc.Array(),
		m.Array(),
		heff.Array(),
		msat0T0.Array(),
		mu.Array(),
		mu.Multiplier())

	llbar_local02nc.Array().Sync()
}
