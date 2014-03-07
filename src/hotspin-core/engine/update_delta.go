//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

import (
	. "hotspin-core/common"
	"hotspin-core/gpu"
)

// Update Δ of quantity
type ΔUpdater struct {
	in  *Quant
	out *Quant
	q0  *Quant
}

func NewΔUpdater(in, ref, out *Quant) *ΔUpdater {
	// look for ground state quantity
	u := new(ΔUpdater)
	u.in = in
	u.out = out
	u.q0 = ref
	engine.Depends(out.Name(), in.Name(), u.q0.Name())
	return u
}

func (u *ΔUpdater) Update() {

	qin := u.in.Array()
	qout := u.out.Array()
	q0 := u.q0.Array()

	qinMul := u.in.Multiplier()
	qoutMul := u.out.Multiplier()
	q0Mul := u.q0.Multiplier()

	COMP := u.in.NComp()
	pre := make([]float64, COMP)

	for ii := 0; ii < COMP; ii++ {
		qoutMul[ii] = qinMul[ii]
		pre[ii] = -q0Mul[ii] / qinMul[ii] //0?
	}

	switch COMP {
	case 1:
		gpu.Madd(qout.Component(0), qin.Component(0), q0.Component(0), pre[0])
	case 3:
		gpu.VecMadd(qout, qin, q0, pre)
	default:
		panic(InputErrF("Δ is not implemented for NComp: ", COMP))
	}

}
