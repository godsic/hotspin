//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// This file implements the externally applied field
// Author: Arne Vansteenkiste

import (
	. "hotspin-core/common"
	. "hotspin-core/engine"
)

// Register this module
func init() {
	RegisterModule("zeeman", "Externally applied magnetic field", LoadZeeman)
}

func LoadZeeman(e *Engine) {
	LoadHField(e)

	e.AddNewQuant("B_ext", VECTOR, MASK, Unit("T"), "externally applied magnetic field")
	hfield := e.Quant("H_eff")
	sum := hfield.Updater().(*SumUpdater)
	sum.MAddParent("B_ext", 1/Mu0)
	e.Depends("H_eff", "B_ext")
}
