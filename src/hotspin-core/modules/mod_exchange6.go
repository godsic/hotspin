//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// 6-neighbor exchange interaction
// Author: Arne Vansteenkiste

import (
	. "hotspin-core/common"
	. "hotspin-core/engine"
	"hotspin-core/gpu"
)

// Register this module
func init() {
	RegisterModule("exchange6", "6-neighbor ferromagnetic exchange interaction", LoadExch6)
}

func LoadExch6(e *Engine) {
	LoadHField(e)
	LoadFullMagnetization(e)
	lex := e.AddNewQuant("lex", SCALAR, MASK, Unit("m"), "Exchange length") // TODO: mask
	Hex := e.AddNewQuant("H_ex", VECTOR, FIELD, Unit("A/m"), "Exchange field")
	hfield := e.Quant("H_eff")
	sum := hfield.Updater().(*SumUpdater)
	sum.AddParent("H_ex")
	e.Depends("H_ex", "lex", "Msat0T0", "m")
	Hex.SetUpdater(&exch6Updater{m: e.Quant("m"), lex: lex, Hex: Hex, Msat0T0: e.Quant("Msat0T0")})
}

type exch6Updater struct {
	m, lex, Hex, Msat0T0 *Quant
}

func (u *exch6Updater) Update() {
	e := GetEngine()
	m := u.m
	lex := u.lex
	Hex := u.Hex
	Msat0T0 := u.Msat0T0

	lexMul2 := lex.Multiplier()[0] * lex.Multiplier()[0]

	lexMul2_cellSizeX2 := lexMul2 / (e.CellSize()[X] * e.CellSize()[X])
	lexMul2_cellSizeY2 := lexMul2 / (e.CellSize()[Y] * e.CellSize()[Y])
	lexMul2_cellSizeZ2 := lexMul2 / (e.CellSize()[Z] * e.CellSize()[Z])

	lexMul2_cellSize2 := []float64{0.0, 0.0, 0.0}
	lexMul2_cellSize2[X] = lexMul2_cellSizeX2
	lexMul2_cellSize2[Y] = lexMul2_cellSizeY2
	lexMul2_cellSize2[Z] = lexMul2_cellSizeZ2

	stream := u.Hex.Array().Stream
	gpu.Exchange6Async(Hex.Array(), m.Array(), Msat0T0.Array(), lex.Array(), Msat0T0.Multiplier()[0], lexMul2_cellSize2, e.Periodic(), stream)
	stream.Sync()
}
