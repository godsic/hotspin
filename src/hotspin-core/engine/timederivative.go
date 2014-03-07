//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Implements the time derivative of a quantity
// Author: Arne Vansteenkiste
// BUG: only works correctly if Update() is called at each time step 
// (which is usually the case).
// TODO: fix

import (
	"math"
	. "hotspin-core/common"
	"hotspin-core/gpu"
)

// TODO: multipliers??
// TODO: not very accurate yet

// Load time derivative of quant if not yet present
func (e *Engine) AddTimeDerivative(q *Quant) {
	panic("//")
	//	name := "d" + q.Name() + "_" + "dt"
	//	if e.HasQuant(name) {
	//		return
	//	}
	//	Assert(q.Kind() == FIELD)
	//	diff := e.AddNewQuant(name, q.NComp(), FIELD, "("+q.Unit()+")/s", "time derivative of "+q.Name())
	//	e.Depends(name, q.Name(), "t", "step")
	//	updater := newDerivativeUpdater(q, diff)
	//	diff.SetUpdater(updater)
	//	diff.SetInvalidator(updater)
}

type derivativeUpdater struct {
	val, diff         *Quant     // original and derived quantities
	lastVal, lastDiff *gpu.Array // previous value for numerical derivative
	lastT             float64    // time of previous value
	lastStep          int        // step of previous value
}

func newDerivativeUpdater(orig, diff *Quant) *derivativeUpdater {
	u := new(derivativeUpdater)
	u.val = orig
	u.diff = diff
	u.lastVal = gpu.NewArray(orig.NComp(), orig.Size3D())  // TODO: alloc only if needed?
	u.lastDiff = gpu.NewArray(orig.NComp(), orig.Size3D()) // TODO: alloc only if needed?
	u.lastT = math.Inf(-1)                                 // so the first time the derivative is taken it will be 0
	u.lastStep = 0                                         //?
	return u
}

func (u *derivativeUpdater) Update() {
	if DEBUG {
		Debug("diff update")
	}

	t := engine.time.Scalar()
	dt := t - u.lastT
	Assert(dt >= 0)
	diff := u.diff.Array()
	val := u.val.Array()
	if dt == 0 {
		if DEBUG {
			Debug("dt==0")
		}
		diff.CopyFromDevice(u.lastDiff)
	} else {
		if DEBUG {
			Debug("dt!=0")
		}
		gpu.LinearCombination2Async(diff, val, float32(1/dt), u.lastVal, -float32(1/dt), diff.Stream)
		diff.Sync()
	}

}

// called when orig, dt or step changes
// TODO: pre-invalidator
func (u *derivativeUpdater) Invalidate() {
	e := GetEngine()
	step := int(e.step.Multiplier()[0])
	if u.lastStep != step {
		if DEBUG {
			Debug("diff invalidate")
		}

		// make sure value is up to date
		u.val.GetUpdater().Update() // TODO: only if needed !!
		u.Update()                  // make sure diff is up to date for lastdiff

		u.lastVal.CopyFromDevice(u.val.Array())
		u.lastDiff.CopyFromDevice(u.diff.Array())
		u.lastT = e.time.Scalar()
		u.lastStep = step
	}
}
