//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// This file implements the uniaxial anisotropy module
// Author: Arne Vansteenkiste

import (
	. "hotspin-core/common"
	. "hotspin-core/engine"
	"hotspin-core/gpu"
)

// Register this module
func init() {
	RegisterModule("anisotropy/uniaxial", "Uniaxial magnetocrystalline anisotropy", LoadAnisUniaxial)
}

func LoadAnisUniaxial(e *Engine) {
	LoadHField(e)
	LoadFullMagnetization(e)

	Hanis := e.AddNewQuant("H_anis", VECTOR, FIELD, Unit("A/m"), "uniaxial anisotropy field")
	ku := e.AddNewQuant("Ku", SCALAR, MASK, Unit("J/m3"), "uniaxial anisotropy constant K")
	anisU := e.AddNewQuant("anisU", VECTOR, MASK, Unit(""), "uniaxial anisotropy direction (unit vector)")

	hfield := e.Quant("H_eff")
	sum := hfield.Updater().(*SumUpdater)
	sum.AddParent("H_anis")
	e.Depends("H_anis", "Ku", "anisU", "msat0T0", "mf")

	Hanis.SetUpdater(&UniaxialAnisUpdater{e.Quant("mf"), Hanis, ku, e.Quant("msat0T0"), anisU})
}

type UniaxialAnisUpdater struct {
	m, hanis, ku, msat, anisU *Quant
}

func (u *UniaxialAnisUpdater) Update() {
	hanis := u.hanis.Array()
	m := u.m.Array()
	ku := u.ku.Array()
	kumul := u.ku.Multiplier()[0]
	anisU := u.anisU.Array()
	anisUMul := u.anisU.Multiplier()
	stream := u.hanis.Array().Stream
	msat := u.msat

	gpu.UniaxialAnisotropyAsync(hanis, m, ku, msat.Array(), 2*kumul/(Mu0*msat.Multiplier()[0]), anisU, anisUMul, stream)

	stream.Sync()
}
