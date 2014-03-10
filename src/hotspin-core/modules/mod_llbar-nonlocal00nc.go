//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// Module implements nonconservative zero-order nonlocal relaxation damping
// Authors: Mykola Dvornik, Arne Vansteenkiste

import (
	. "hotspin-core/common"
	. "hotspin-core/engine"
	"hotspin-core/gpu"
)

// Register this module
func init() {
	RegisterModule("llbar/damping/nonconservative/00/nonlocal", "LLBar nonconservative zero-order nonlocal relaxation term", LoadLLBarNonlocal00NC)
}

func LoadLLBarNonlocal00NC(e *Engine) {

	LoadHField(e)
	LoadFullMagnetization(e)
	LoadGammaLL(e)

	// ============ New Quantities =============
	e.AddNewQuant("λₑ∥", VECTOR, MASK, Unit(""), "LLBar zero-order non-local relaxation diagonal tensor")
	llbar_nonlocal00nc := e.AddNewQuant("llbar_nonlocal00nc", VECTOR, FIELD, Unit("/s"), "Landau-Lifshitz-Baryakhtar nonconservative zero-order nonlocal relaxation term")

	// ============ Dependencies =============
	e.Depends("llbar_nonlocal00nc", "H_eff", "γ_LL", "λₑ∥", "msat0T0")

	// ============ Updating the torque =============
	upd := &LLBarNonlocal00NCUpdater{llbar_nonlocal00nc: llbar_nonlocal00nc}
	llbar_nonlocal00nc.SetUpdater(upd)
}

type LLBarNonlocal00NCUpdater struct {
	llbar_nonlocal00nc *Quant
}

func (u *LLBarNonlocal00NCUpdater) Update() {

	e := GetEngine()
	llbar_nonlocal00nc := u.llbar_nonlocal00nc
	gammaLL := e.Quant("γ_LL").Scalar()
	cellSize := e.CellSize()
	heff := e.Quant("H_eff")
	pbc := e.Periodic()

	// put gamma in multiplier to avoid additional multiplications
	multiplierBT := llbar_nonlocal00nc.Multiplier()
	for i := range multiplierBT {
		multiplierBT[i] = gammaLL
	}

	lambda_e := e.Quant("λₑ∥")
	msat0T0 := e.Quant("msat0T0")

	gpu.LLBarNonlocal00NC(llbar_nonlocal00nc.Array(),
		heff.Array(),
		msat0T0.Array(),
		lambda_e.Array(),
		lambda_e.Multiplier(),
		float64(cellSize[X]),
		float64(cellSize[Y]),
		float64(cellSize[Z]),
		pbc)

	llbar_nonlocal00nc.Array().Sync()
}
