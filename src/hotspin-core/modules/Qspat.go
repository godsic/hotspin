//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

import (
	. "hotspin-core/engine"
	"hotspin-core/gpu"
)

func LoadQspat(e *Engine, tName string, fName string, cName string) {

	LoadTemp(e, tName)

	k := e.AddNewQuant(cName, SCALAR, MASK, Unit("J/(s*K*m)"), "Heat conductivity")
	Qspat := e.AddNewQuant(fName, SCALAR, FIELD, Unit("J/(s*m3)"), "Heat flux density caused by spatial temperature gradient")
	T := e.Quant(tName)
	e.Depends(fName, cName, tName)
	Qspat.SetUpdater(&QspatUpdater{Qspat: Qspat, T: T, k: k})
}

type QspatUpdater struct {
	Qspat, T, k *Quant
}

func (u *QspatUpdater) Update() {
	e := GetEngine()
	pbc := e.Periodic()
	cellSize := e.CellSize()

	gpu.Qspat_async(
		u.Qspat.Array(),
		u.T.Array(),
		u.k.Array(),
		u.k.Multiplier(),
		cellSize,
		pbc)

	u.Qspat.Array().Sync()
}
