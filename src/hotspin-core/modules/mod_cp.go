//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// temperature dependence of longitudinal susceptibility as follows from mean-field approximation
// Author: Mykola Dvornik

import (
	. "hotspin-core/common"
	. "hotspin-core/engine"
	"hotspin-core/gpu"
)

var inCp = map[string]string{
	"T": LtempName,
}

var depsCp = map[string]string{
	"Td": "Td",
	"n":  "n",
}

var outCp = map[string]string{
	LcapacName: LcapacName,
}

// Register this module
func init() {
	args := Arguments{inCp, depsCp, outCp}
	RegisterModuleArgs("temperature/Debye", "Phonons specific heat according to Debye model", args, LoadDebyeCpArgs)
}

func LoadDebyeCpArgs(e *Engine, args ...Arguments) {

	// make it automatic !!!
	var arg Arguments

	if len(args) == 0 {
		arg = Arguments{inCp, depsCp, outCp}
	} else {
		arg = args[0]
	}

	LoadTemp(e, arg.Ins("T"))
	LoadMFAParams(e)

	if !e.HasQuant(arg.Outs(LcapacName)) {
		e.AddNewQuant(arg.Outs(LcapacName), SCALAR, FIELD, Unit("J/(K*m3)"), "The volumetric heat capacity of the thermal bath")
	}

	e.AddNewQuant(arg.Deps("Td"), SCALAR, MASK, Unit("K"), "Debye temperature")

	e.Depends(arg.Outs(LcapacName), arg.Deps("Td"), arg.Deps("n"), arg.Ins("T"))

	T := e.Quant(arg.Ins("T"))
	n := e.Quant(arg.Deps("n"))
	Td := e.Quant(arg.Deps("Td"))
	Cp := e.Quant(arg.Outs(LcapacName))

	Cp.SetUpdater(&CpDebyeUpdater{Cp: Cp, T: T, Td: Td, n: n})

}

type CpDebyeUpdater struct {
	Cp, T, Td, n *Quant
}

func (u *CpDebyeUpdater) Update() {
	Cp := u.Cp
	T := u.T
	Td := u.Td
	n := u.n

	stream := Cp.Array().Stream
	Cp.Multiplier()[0] = 9.0 * n.Multiplier()[0] * Kb

	gpu.CpAsync(Cp.Array(), T.Array(), Td.Array(), n.Array(), Td.Multiplier()[0], stream)
	stream.Sync()
}
