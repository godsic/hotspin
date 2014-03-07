//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Author: Arne Vansteenkiste

import (
	. "hotspin-core/common"
	"hotspin-core/gpu"
)

type HeunSolver struct {
	buffer []*gpu.Array
}

func NewHeun(e *Engine) *HeunSolver {
	s := new(HeunSolver)
	s.buffer = make([]*gpu.Array, len(e.equation))
	return s
}

func (s *HeunSolver) Step() {
	e := GetEngine()
	equation := e.equation

	// First update all inputs
	dt := engine.dt.Scalar()
	for i := range equation {
		Assert(equation[i].kind == EQN_PDE1)
		equation[i].input[0].Update()
	}

	// Then step all outputs
	// and invalidate them.

	// stage 0
	for i := range equation {
		y := equation[i].output[0]
		dy := equation[i].input[0]
		dyMul := dy.multiplier
		checkUniform(dyMul)
		s.buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.buffer[i].CopyFromDevice(dy.Array()) // save for later

		gpu.Madd(y.Array(), y.Array(), dy.Array(), dt*dyMul[0]) // initial euler step

		y.Invalidate()
	}

	// Advance time
	e.time.SetScalar(e.time.Scalar() + dt)

	// update inputs again
	for i := range equation {
		Assert(equation[i].kind == EQN_PDE1)
		equation[i].input[0].Update()
	}

	// stage 1
	for i := range equation {
		y := equation[i].output[0]
		dy := equation[i].input[0]
		dyMul := dy.multiplier

		h := float32(dt * dyMul[0])
		gpu.MAdd2Async(y.Array(), dy.Array(), 0.5*h, s.buffer[i], -0.5*h, y.Array().Stream) // corrected step
		y.Array().Sync()
		Pool.Recycle(&s.buffer[i])

		y.Invalidate()
	}

	e.step.SetScalar(e.step.Scalar() + 1) // advance time step
}
