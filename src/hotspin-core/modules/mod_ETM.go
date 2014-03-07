//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

import (
	//~ . "hotspin-core/common"
	. "hotspin-core/engine"
)

const EfluxName = "Qe"
const EspatFluxName = "Qe_spat"
const EtempName = "Te"
const ErateName = "dTeds"
const EcapacName = "Cp_e"
const EcondName = "k_e"
const EpowName = "pow_e"

const Epow = 1.0

// Register this module
func init() {
	RegisterModule("temperature/ETM", "Electron temperature model", LoadETM)
}

func LoadETM(e *Engine) {
	LoadTM(e, EtempName, EfluxName, ErateName, EcapacName, EpowName, Epow)
}
