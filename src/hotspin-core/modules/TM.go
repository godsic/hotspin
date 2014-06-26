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

func LoadTM(e *Engine, tName string, fName string, rName string, cName string) {

	// Load temperature quantity
	LoadTemp(e, tName)
	// Load heat flux density quantity
	LoadQ(e, fName)

	dTds := e.AddNewQuant(rName, SCALAR, FIELD, Unit("K/s"), "The temperature change rate")

	if !e.HasQuant(cName) {
		e.AddNewQuant(cName, SCALAR, MASK, Unit("J/(K*m3)"), "The volumetric heat capacity of the thermal bath")
	}

	Q := e.Quant(fName)
	T := e.Quant(tName)
	Cp := e.Quant(cName)

	dTds.SetUpdater(&dTdsUpdater{dTds: dTds, Q: Q, Cp: Cp, T: T})
	e.Depends(rName, fName, cName)
	e.AddPDE1(tName, rName)
}

type dTdsUpdater struct {
	dTds, Q, Cp, T *Quant
}

func (u *dTdsUpdater) Update() {
	nCpn := u.Cp.Multiplier()[0]

	var pre float64
	if nCpn != 0.0 {
		pre = 1.0 / nCpn
	} else {
		pre = 0.0
	}

	mult := u.dTds.Multiplier()
	for i := range mult {
		mult[i] = pre
	}

	gpu.Div(u.dTds.Array(),
		u.Q.Array(),
		u.Cp.Array())
}
