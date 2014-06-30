//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Module implements conservative second-order local damping
// Authors: Mykola Dvornik

import (
	. "hotspin-core/engine"
	"hotspin-core/gpu"
)

// Register this module
func init() {
	RegisterModule("llbar/damping/conservative/02/local", "LLBar conservative second-order local relaxation term", LoadLLBarLocal02C)
}

func LoadLLBarLocal02C(e *Engine) {

	LoadHField(e)
	LoadFullMagnetization(e)
	LoadGammaLL(e)

	// =========== New Quantities =============

	e.AddNewQuant("μ⊥", VECTOR, MASK, Unit(""), "LLBar second-order local relaxation diagonal tensor")
	llbar_local02c := e.AddNewQuant("llbar_local02c", VECTOR, FIELD, Unit("/s"), "Landau-Lifshitz-Baryakhtar conservative second-order local relaxation term")

	// ============ Dependencies =============
	e.Depends("llbar_local02c", "m", "H_eff", "γ_LL", "μ⊥", "msat0T0")

	// ============ Updating the torque =============
	upd := &LLBarLocal02CUpdater{llbar_local02c: llbar_local02c}
	llbar_local02c.SetUpdater(upd)
}

type LLBarLocal02CUpdater struct {
	llbar_local02c *Quant
}

func (u *LLBarLocal02CUpdater) Update() {

	e := GetEngine()

	llbar_local02c := u.llbar_local02c
	gammaLL := e.Quant("γ_LL").Scalar()
	m := e.Quant("m") // m is M/Ms(T=0)
	heff := e.Quant("H_eff")

	// put gamma in multiplier to avoid additional multiplications
	multiplierBT := llbar_local02c.Multiplier()
	for i := range multiplierBT {
		multiplierBT[i] = gammaLL
	}

	mu := e.Quant("μ⊥")
	msat0T0 := e.Quant("msat0T0")

	gpu.LLBarLocal02C(llbar_local02c.Array(),
		m.Array(),
		heff.Array(),
		msat0T0.Array(),
		mu.Array(),
		mu.Multiplier())

	llbar_local02c.Array().Sync()
}
