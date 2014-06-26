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

var inCe = map[string]string{
	"T": EtempName,
}

var depsCe = map[string]string{
	"n": "n",
	"γ": "γ",
}

var outCe = map[string]string{
	EcapacName: EcapacName,
}

// Register this module
func init() {
	args := Arguments{inCe, depsCe, outCe}
	RegisterModuleArgs("temperature/Drude", "Electrons specific heat according to Drude model", args, LoadCeDrudeArgs)
}

func LoadCeDrudeArgs(e *Engine, args ...Arguments) {

	// make it automatic !!!
	var arg Arguments

	if len(args) == 0 {
		arg = Arguments{inCe, depsCe, outCe}
	} else {
		arg = args[0]
	}

	LoadTemp(e, arg.Ins("T"))
	LoadMFAParams(e)

	if !e.HasQuant(arg.Outs(EcapacName)) {
		e.AddNewQuant(arg.Outs(EcapacName), SCALAR, FIELD, Unit("J/(K*m3)"), "The volumetric heat capacity of the electrons")
	}

	e.AddNewQuant(arg.Deps("γ"), SCALAR, MASK, Unit("J/(K^2*mol)"), "Electron specific heat constant")

	e.Depends(arg.Outs(EcapacName), arg.Deps("γ"), arg.Deps("n"), arg.Ins("T"))

	T := e.Quant(arg.Ins("T"))
	n := e.Quant(arg.Deps("n"))
	γ := e.Quant(arg.Deps("γ"))
	Ce := e.Quant(arg.Outs(EcapacName))

	Ce.SetUpdater(&CeDrudeUpdater{Ce: Ce, T: T, γ: γ, n: n})

}

type CeDrudeUpdater struct {
	Ce, T, γ, n *Quant
}

func (u *CeDrudeUpdater) Update() {
	Ce := u.Ce
	T := u.T
	γ := u.γ
	n := u.n

	stream := Ce.Array().Stream

	Ce.Multiplier()[0] = (n.Multiplier()[0] / Na) * γ.Multiplier()[0]

	Ce.Array().CopyFromDevice(T.Array())

	if !n.Array().IsNil() {
		gpu.Mul(Ce.Array(), n.Array(), Ce.Array())
	}

	if !γ.Array().IsNil() {
		gpu.Mul(Ce.Array(), γ.Array(), Ce.Array())
	}

	stream.Sync()
}
