//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

import (
	"math"
	//~ . "hotspin-core/common"
	. "hotspin-core/engine"
	"hotspin-core/gpu"
)

func LoadTM(e *Engine, tName string, fName string, rName string, cName string, pName string, pDefVal float64) {

	// Load temperature quantity
	LoadTemp(e, tName)
	// Load heat flux density quantity
	LoadQ(e, fName)

	dTds := e.AddNewQuant(rName, SCALAR, FIELD, Unit("K/s"), "The temperature change rate")

	if !e.HasQuant(cName) {
		e.AddNewQuant(cName, SCALAR, MASK, Unit("J/(K*m3)"), "The volumetric heat capacity of the thermal bath")
	}

	Pow := e.AddNewQuant(pName, SCALAR, VALUE, Unit(""), "Slope of temperature dependence of specific heat capacity")
	Pow.SetScalar(pDefVal)

	Q := e.Quant(fName)
	T := e.Quant(tName)
	Cp := e.Quant(cName)

	dTds.SetUpdater(&dTdsUpdater{dTds: dTds, Q: Q, Cp: Cp, T: T, Pow: Pow})
	if pDefVal == 0.0 {
		e.Depends(rName, fName, cName)
	} else {
		e.Depends(rName, fName, cName, tName, pName)
	}
	e.AddPDE1(tName, rName)
}

type dTdsUpdater struct {
	dTds, Q, Cp, T, Pow *Quant
}

func (u *dTdsUpdater) Update() {
	nCpn := u.Cp.Multiplier()[0]
	pow := -u.Pow.Scalar()

	var pre float64
	if nCpn != 0.0 {
		pre = 1.0 / nCpn
	} else {
		pre = 0.0
	}
	if pow != 0.0 {
		pre = pre * math.Pow(u.T.Multiplier()[0], pow)
	}

	mult := u.dTds.Multiplier()
	for i := range mult {
		mult[i] = pre
	}
	if pow == 0.0 {
		gpu.Div(u.dTds.Array(),
			u.Q.Array(),
			u.Cp.Array())
	} else {
		gpu.DivMulPow(u.dTds.Array(),
			u.Q.Array(),
			u.Cp.Array(),
			u.T.Array(),
			pow)
	}
}
