//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// This file implements micromagnetic energy terms
// Author: Arne Vansteenkiste

import (
	. "hotspin-core/common"
	. "hotspin-core/engine"
	"hotspin-core/gpu"
	"strings"
)

// Register this module
func init() {
	RegisterModule("micromag/energy", "Micromagnetic energy terms **of fields loaded before this module**.", LoadEnergy)
}

func LoadEnergy(e *Engine) {
	LoadHField(e)
	LoadFullMagnetization(e)

	M := "mf"

	total := e.AddNewQuant("E", SCALAR, VALUE, Unit("J"), "Sum of all calculated energy terms (this is the total energy only if all relevant energy terms are loaded")
	sumUpd := NewSumUpdater(total).(*SumUpdater)
	total.SetUpdater(sumUpd)

	if e.HasQuant("B_ext") {
		term := LoadEnergyTerm(e, "E_zeeman", M, "B_ext", -e.CellVolume(), "Zeeman energy")
		Log("Loaded Zeeman energy E_zeeman")
		sumUpd.AddParent(term.Name())
	}

	if e.HasQuant("H_ex") {
		term := LoadEnergyTerm(e, "E_ex", M, "H_ex", -0.5*e.CellVolume()*Mu0, "Exchange energy")
		Log("Loaded exchange energy E_ex")
		sumUpd.AddParent(term.Name())
	}

	// WARNING: this assumes B is only B_demag.
	if e.HasQuant("B") {
		term := LoadEnergyTerm(e, "E_demag", M, "B", -0.5*e.CellVolume(), "Demag energy")
		Log("Loaded demag energy E_demag")
		sumUpd.AddParent(term.Name())
	}

	if e.HasQuant("H_anis") {
		term := LoadEnergyTerm(e, "E_anis", M, "H_anis", -0.5*e.CellVolume()*Mu0, "Anisotropy energy")
		Log("Loaded anisotropy energy E_anis")
		sumUpd.AddParent(term.Name())
	}

	if e.HasQuant("H_lf") {
		term := LoadEnergyTerm(e, "E_lf", M, "H_lf", -e.CellVolume()*Mu0, "Longitudinal field energy")
		Log("Loaded anisotropy energy E_lf")
		sumUpd.AddParent(term.Name())
	}
}

func LoadEnergyTerm(e *Engine, out, in1, in2 string, weight float64, desc string) *Quant {
	Energy := e.AddNewQuant(out, SCALAR, VALUE, Unit("J"), desc)
	EnergyDensityName := strings.Replace(out, "E_", "w_", -1)
	EnergyDensity := e.AddNewQuant(EnergyDensityName, SCALAR, FIELD, Unit("J/m3"), desc)
	e.Depends(out, in1, in2)
	e.Depends(EnergyDensityName, in1, in2)
	m := e.Quant(in1)
	H := e.Quant(in2)
	Energy.SetUpdater(NewEnergyUpdater(Energy, EnergyDensity, m, H, e.Quant("msat0T0"), weight))
	return Energy
}

type EnergyUpdater struct {
	*SumReduceUpdater
	energy *Quant
	msat   *Quant
	m      *Quant
	H      *Quant
	w      *Quant
	weight float64
}

func NewEnergyUpdater(result, w, m, H, msat *Quant, weight float64) Updater {
	u := new(EnergyUpdater)
	u.SumReduceUpdater = NewSumReduceUpdater(w, result).(*SumReduceUpdater)
	u.energy = result
	u.w = w
	u.m = m
	u.H = H
	u.msat = msat
	u.weight = weight
	return u
}

func (u *EnergyUpdater) Update() {

	gpu.DotMask(u.w.Array(), u.m.Array(), u.H.Array(), u.m.Multiplier(), u.H.Multiplier())

	if !u.msat.Array().IsNil() {
		gpu.Mul(u.w.Array(), u.w.Array(), u.msat.Array())
	}

	u.SumReduceUpdater.Update()

	u.energy.Multiplier()[0] *= u.msat.Multiplier()[0]
	u.energy.Multiplier()[0] *= u.weight

	u.energy.SetUpToDate(true)
	u.w.SetUpToDate(true)
}
