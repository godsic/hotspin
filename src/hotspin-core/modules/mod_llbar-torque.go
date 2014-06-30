//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Module implementes LLBar torque (indestiguishible from LL)
// Authors: Mykola Dvornik, Arne Vansteenkiste

import (
	. "hotspin-core/engine"
	"hotspin-core/gpu"
)

// Register this module
func init() {
	RegisterModule("llbar/torque", "LLBar torque term", LoadLLBarTorque)
}

func LoadLLBarTorque(e *Engine) {

	LoadHField(e)
	LoadFullMagnetization(e)
	LoadGammaLL(e)

	// ============ New Quantities =============

	llbar_torque := e.AddNewQuant("llbar_torque", VECTOR, FIELD, Unit("/s"), "Landau-Lifshitz-Baryakhtar torque")

	// ============ Dependencies =============
	e.Depends("llbar_torque", "m", "H_eff", "γ_LL")

	// ============ Updating the torque =============
	upd := &LLBarTorqueUpdater{llbar_torque: llbar_torque}
	llbar_torque.SetUpdater(upd)
}

type LLBarTorqueUpdater struct {
	llbar_torque *Quant
}

func (u *LLBarTorqueUpdater) Update() {

	e := GetEngine()
	llbar_torque := u.llbar_torque
	gammaLL := e.Quant("γ_LL").Scalar()
	m := e.Quant("m")
	heff := e.Quant("H_eff")

	// put gamma in multiplier to avoid additional multiplications
	multiplierBT := llbar_torque.Multiplier()

	for i := range multiplierBT {
		multiplierBT[i] = gammaLL
	}

	msat0T0 := e.Quant("msat0T0")

	gpu.LLBarTorqueAsync(llbar_torque.Array(),
		m.Array(),
		heff.Array(),
		msat0T0.Array())

	llbar_torque.Array().Sync()
}
