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

func LoadQinter(e *Engine, fName string, TiName string, TjName string, GijName string) {

	LoadTemp(e, TiName)
	LoadTemp(e, TjName)

	Gij := e.AddNewQuant(GijName, SCALAR, MASK, Unit("J/(s*K*m3)"), "Coupling constant")
	Qinter := e.AddNewQuant(fName, SCALAR, FIELD, Unit("J/(s*m3)"), "Heat flux density caused by spatial temperature gradient")
	Ti := e.Quant(TiName)
	Tj := e.Quant(TjName)
	e.Depends(fName, TiName, TjName, GijName)
	Qinter.SetUpdater(&QinterUpdater{Qinter: Qinter, Ti: Ti, Tj: Tj, Gij: Gij})
}

type QinterUpdater struct {
	Qinter, Ti, Tj, Gij *Quant
}

func (u *QinterUpdater) Update() {

	stream := u.Qinter.Array().Stream
	gpu.Qinter_async(
		u.Qinter.Array(),
		u.Ti.Array(),
		u.Tj.Array(),
		u.Gij.Array(),
		u.Gij.Multiplier(),
		stream)
	stream.Sync()
}
