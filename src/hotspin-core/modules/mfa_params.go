//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// File provides all the quantities within MFA.
// Author: Mykola Dvornik

import (
	//~ . "hotspin-core/common"
	. "hotspin-core/engine"
)

// Loads the "Q" quantity if it is not already present.
func LoadMFAParams(e *Engine) {
	if !e.HasQuant("Tc") {
		e.AddNewQuant("Tc", SCALAR, MASK, Unit("K"), "Curie temperature")
	}
	if !e.HasQuant("J") {
		e.AddNewQuant("J", SCALAR, MASK, Unit(""), "Full atomic angular momentum")
	}

	if !e.HasQuant("n") {
		e.AddNewQuant("n", SCALAR, MASK, Unit("1/m3"), "Number of spins in the unit volume")
	}
}
