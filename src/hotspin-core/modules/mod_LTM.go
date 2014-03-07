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

const LfluxName = "Ql"
const LspatFluxName = "Ql_spat"
const LtempName = "Tl"
const LrateName = "dTlds"
const LcapacName = "Cp_l"
const LcondName = "k_l"
const LpowName = "pow_l"

const Lpow = 0.0

// Register this module
func init() {
	RegisterModule("temperature/LTM", "Lattice temperature model", LoadLTM)
}

func LoadLTM(e *Engine) {
	LoadTM(e, LtempName, LfluxName, LrateName, LcapacName, LpowName, Lpow)
}
