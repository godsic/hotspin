//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Author: Mykola Dvornik, Arne Vansteenkiste

import (
	"fmt"
	. "hotspin-core/common"
	"hotspin-core/gpu"
)

// Naive Backward Euler solver
type BDFEuler struct {
	ybuffer    []*gpu.Array // initial derivative
	y0buffer   []*gpu.Array // initial derivative
	err        []*Quant     // error estimates for each equation
	maxIterErr []*Quant     // error estimates for each equation
	maxIter    []*Quant     // maximum number of iterations per step
	diff       []gpu.Reductor
	iterations *Quant
}

func (s *BDFEuler) Step() {
	e := GetEngine()

	equation := e.equation

	for i := range equation {
		equation[i].input[0].Update()
	}

	// get dt here to avoid updates later on.
	dt := engine.dt.Scalar()
	// Advance time and update all inputs
	e.time.SetScalar(e.time.Scalar() + dt)

	// Then step all outputs (without intermediate updates!)
	// and invalidate them.

	for i := range equation {
		err := 1.0e38
		s.iterations.SetScalar(0)
		iter := 0

		// Do forward Euler step
		// Zero order approximation
		y := equation[i].output[0]
		dy := equation[i].input[0]
		dyMul := dy.multiplier
		h := dt * dyMul[0]
		s.y0buffer[i].CopyFromDevice(y.Array()) // save for later
		gpu.Madd(y.Array(), s.y0buffer[i], dy.Array(), h)
		y.Invalidate()
		equation[i].input[0].Update()
		s.iterations.SetScalar(s.iterations.Scalar() + 1)

		// Do backward Euler step and solve it
		// Do higher order approximation until converges
		// Using fixed-point iterator

		maxIterErr := s.maxIterErr[i].Scalar()
		maxIter := int(s.maxIter[i].Scalar())

		for err > maxIterErr {
			gpu.Madd(s.ybuffer[i], s.y0buffer[i], dy.Array(), h)
			err = float64(s.diff[i].MaxDiff(y.Array(), s.ybuffer[i]))
			sum := float64(s.diff[i].MaxSum(y.Array(), s.ybuffer[i]))
			if sum > 0.0 {
				err = err / sum
			}
			iter = iter + 1
			s.iterations.SetScalar(s.iterations.Scalar() + 1)
			y.Array().CopyFromDevice(s.ybuffer[i])
			y.Invalidate()
			equation[i].input[0].Update()
			if iter > maxIter {
				panic(Bug(fmt.Sprintf("The BDF iterator cannot converge for %s! Please decrease the time step and re-run!", y.Name())))
			}
		}

	}

	// Advance step
	e.step.SetScalar(e.step.Scalar() + 1) // advance step
}

func (s *BDFEuler) Dependencies() (children, parents []string) {
	children = []string{"t", "step", "bdf_iterations"}
	parents = []string{"dt"}
	for i := range s.err {
		parents = append(parents, s.maxIter[i].Name())
		parents = append(parents, s.maxIterErr[i].Name())
	}
	return
}

// Register this module
func init() {
	RegisterModule("solver/bdf-euler", "Fixed-step Backward Euler solver", LoadBDFEuler)
}

func LoadBDFEuler(e *Engine) {
	s := new(BDFEuler)
	s.iterations = e.AddNewQuant("bdf_iterations", SCALAR, VALUE, Unit(""), "Number of iterations per step")

	equation := e.equation
	s.ybuffer = make([]*gpu.Array, len(equation))
	s.y0buffer = make([]*gpu.Array, len(equation))

	s.err = make([]*Quant, len(equation))
	s.diff = make([]gpu.Reductor, len(equation))
	s.maxIter = make([]*Quant, len(equation))
	s.maxIterErr = make([]*Quant, len(equation))

	for i := range equation {

		eqn := &(equation[i])
		Assert(eqn.kind == EQN_PDE1)
		out := eqn.output[0]
		unit := out.Unit()
		s.err[i] = e.AddNewQuant(out.Name()+"_error", SCALAR, VALUE, unit, "Error/step estimate for "+out.Name())
		s.maxIter[i] = e.AddNewQuant(out.Name()+"_maxIterations", SCALAR, VALUE, unit, "Maximum number of evaluations per step"+out.Name())
		s.maxIterErr[i] = e.AddNewQuant(out.Name()+"_maxIterError", SCALAR, VALUE, unit, "The maximum error of iterator"+out.Name())

		s.maxIterErr[i].SetScalar(1e-5)
		s.maxIter[i].SetScalar(3)

		s.diff[i].Init(out.Array().NComp(), out.Array().Size3D())
		s.maxIter[i].SetVerifier(Positive)
		s.maxIterErr[i].SetVerifier(Positive)

		// TODO: recycle?

		y := equation[i].output[0]
		s.ybuffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.y0buffer[i] = Pool.Get(y.NComp(), y.Size3D())

	}
	e.SetSolver(s)
}
