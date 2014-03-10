//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// This file implements an adaptive Euler-Heun scheme
// Author: Arne Vansteenkiste

import (
	"math"
	. "hotspin-core/common"
	"hotspin-core/gpu"
	"fmt"
)

type RK12Solver struct {
	y0buffer []*gpu.Array // initial value
	dybuffer []*gpu.Array // initial derivative
	err      []*Quant     // error estimates for each equation
	peakErr  []*Quant     // maximum error for each equation
	maxErr   []*Quant     // maximum error for each equation
	diff     []gpu.Reductor
	minDt    *Quant
	maxDt    *Quant
	badSteps *Quant
}

// Load the solver into the Engine
func LoadRK12(e *Engine) {
	s := new(RK12Solver)

	// Minimum/maximum time step
	s.minDt = e.AddNewQuant("mindt", SCALAR, VALUE, Unit("s"), "Minimum time step")
	s.minDt.SetScalar(1e-38)
	s.minDt.SetVerifier(Positive)
	s.maxDt = e.AddNewQuant("maxdt", SCALAR, VALUE, Unit("s"), "Maximum time step")
	s.maxDt.SetVerifier(Positive)
	s.maxDt.SetScalar(1e38)
	s.badSteps = e.AddNewQuant("badsteps", SCALAR, VALUE, Unit(""), "Number of time steps that had to be re-done")

	equation := e.equation
	s.dybuffer = make([]*gpu.Array, len(equation))
	s.y0buffer = make([]*gpu.Array, len(equation))
	s.err = make([]*Quant, len(equation))
	s.peakErr = make([]*Quant, len(equation))
	s.maxErr = make([]*Quant, len(equation))
	s.diff = make([]gpu.Reductor, len(equation))
	e.SetSolver(s)

	for i := range equation {

		eqn := &(equation[i])
		Assert(eqn.kind == EQN_PDE1)
		out := eqn.output[0]
		unit := out.Unit()
		s.err[i] = e.AddNewQuant(out.Name()+"_error", SCALAR, VALUE, unit, "Error/step estimate for "+out.Name())
		s.peakErr[i] = e.AddNewQuant(out.Name()+"_peakerror", SCALAR, VALUE, unit, "All-time maximum error/step for "+out.Name())
		s.maxErr[i] = e.AddNewQuant(out.Name()+"_maxError", SCALAR, VALUE, unit, "Maximum error/step for "+out.Name())
		s.diff[i].Init(out.Array().NComp(), out.Array().Size3D())
		s.maxErr[i].SetVerifier(Positive)

		// TODO: recycle?
		y := equation[i].output[0]
		s.dybuffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.y0buffer[i] = Pool.Get(y.NComp(), y.Size3D())

	}
}

// Declares this solver's special dependencies
func (s *RK12Solver) Dependencies() (children, parents []string) {
	children = []string{"dt", "step", "t", "badsteps"}
	parents = []string{"dt", "mindt", "maxdt"}
	for i := range s.err {
		parents = append(parents, s.maxErr[i].Name())
		children = append(children, s.peakErr[i].Name(), s.err[i].Name())
	}
	return
}

// Register this module
func init() {
	RegisterModule("solver/rk12", "Adaptive Heun solver (Runge-Kutta 1+2)", LoadRK12)
}

// Take one time step
func (s *RK12Solver) Step() {
	e := GetEngine()
	equation := e.equation

	// First update all inputs

	for i := range equation {
		Assert(equation[i].kind == EQN_PDE1)
		equation[i].input[0].Update()
	}

	// Then step all outputs
	// and invalidate them.

	// stage 0
	t0 := e.time.Scalar()
	for i := range equation {
		y := equation[i].output[0]
		dy := equation[i].input[0]
		dyMul := dy.multiplier
		checkUniform(dyMul)
		s.dybuffer[i].CopyFromDevice(dy.Array()) // save for later
		s.y0buffer[i].CopyFromDevice(y.Array()) // save for later

	}

	const maxTry = 10 // undo at most this many bad steps
	const headRoom = 0.8
	try := 0 
	
	for {
		// We need to update timestep if the step has failed
		dt := engine.dt.Scalar()
		// initial euler step
		for i := range equation {
			y := equation[i].output[0]
			dy := equation[i].input[0]
			dyMul := dy.multiplier
			if try > 0 { // restore previous initial conditions
				y.Array().CopyFromDevice(s.y0buffer[i])
				dy.Array().CopyFromDevice(s.dybuffer[i])
			}
			gpu.Madd(y.Array(), y.Array(), dy.Array(), dt*dyMul[0])
			y.Invalidate()
		}

		// Advance time
		e.time.SetScalar(t0 + dt)

		// update inputs again
		for i := range equation {
			equation[i].input[0].Update()
		}

		// stage 1
		badStep := false
		minFactor := 2.0
		for i := range equation {
			y := equation[i].output[0]
			dy := equation[i].input[0]
			dyMul := dy.multiplier

			h := float64(dt * dyMul[0])
			gpu.MAdd2Async(y.Array(), dy.Array(), 0.5*h, s.dybuffer[i], -0.5*h, y.Array().Stream) // corrected step
			y.Array().Sync()

			// error estimate
			stepDiff := s.diff[i].MaxDiff(dy.Array(), s.dybuffer[i]) * h
			err := float64(stepDiff)
			s.err[i].SetScalar(err)
			maxErr := s.maxErr[i].Scalar()
			if err > maxErr {
				s.badSteps.SetScalar(s.badSteps.Scalar() + 1)
				badStep = true
			}
			if (!badStep || try == maxTry-1) && err > s.peakErr[i].Scalar() {
				// peak error should be that of good step, unless last trial which will not be undone
				s.peakErr[i].SetScalar(err)
			}
			factor := 0.0
			if !badStep {
				factor = math.Sqrt(maxErr / err) //maxErr / err
			} else {
				factor = math.Pow(maxErr/err, 1./3.)
			}
			factor *= headRoom
			// do not increase/cut too much
			// TODO: give user the control:
			if factor > 1.5 {
				factor = 1.5
			}
			if factor < 0.1 {
				factor = 0.1
			}
			if factor < minFactor {
				minFactor = factor
			} // take minimum time increase factor of all eqns.

			y.Invalidate()
			//if badStep{break} // do not waste time on other equations
		}

		// Set new time step but do not go beyond min/max bounds
		newDt := dt * minFactor
		if newDt < s.minDt.Scalar() {
			newDt = s.minDt.Scalar()
		}
		if newDt > s.maxDt.Scalar() {
			newDt = s.maxDt.Scalar()
		}
		e.dt.SetScalar(newDt)
		if !badStep || newDt == s.minDt.Scalar() {
			break
		}
		if try > maxTry {
			panic(Bug(fmt.Sprint("The solver cannot converge after ",maxTry," badsteps")))
		}
		
		try++
	} // end try

	// advance time step
	e.step.SetScalar(e.step.Scalar() + 1)
}
