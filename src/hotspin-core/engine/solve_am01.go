//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Author: Mykola Dvornik, Arne Vansteenkiste
// The time stepper is based on "Adaptive Stepsize Control in Implicit Runge-Kutta Methods for Reservoir Simulation" by Carsten V ̈olcker et al.

import (
	"container/list"
	"fmt"
	"math"
	. "hotspin-core/common"
	"hotspin-core/gpu"
	"sort"
)

// The adaptive implicit method, predictor: Implicit Euler, corrector: Trapezoidal
type BDFAM12 struct {
	ybuffer []*gpu.Array // current value of the quantity

	y0buffer []*gpu.Array // the value of quantity at the begining of the step
	y1buffer []*gpu.Array // the value of quantity after pedictor step

	dy0buffer  []*gpu.Array // the value of quantity derivative at the begining of the step
	dybuffer   []*gpu.Array // the buffer for quantity derivative
	err        []*Quant     // error estimates for each equation
	alpha      []float64     // convergence estimates for each equation
	maxAbsErr  []*Quant     // maximum absolute error per step for each equation
	maxRelErr  []*Quant     // maximum absolute error per step for each equation
	newDt      []float64    //
	diff       []gpu.Reductor
	err_list   []*list.List
	steps_list []*list.List
	iterations *Quant
	badSteps   *Quant
	minDt      *Quant
	maxDt      *Quant
}

func (s *BDFAM12) Step() {
	e := GetEngine()
	t0 := e.time.Scalar()

	s.badSteps.SetScalar(0)
	s.iterations.SetScalar(0)

	equation := e.equation
	//~ make sure that errors history is wiped for t0 = 0 s!
	if t0 == 0.0 {
		for i := range equation {
			s.err_list[i].Init()
			s.steps_list[i].Init()
		}
	}
	//~ save everything in the begining
	e.UpdateEqRHS()
	for i := range equation {
		y := equation[i].LHS()
		dy := equation[i].RHS()
		s.y0buffer[i].CopyFromDevice(y.Array())   //~ save for later
		s.dy0buffer[i].CopyFromDevice(dy.Array()) //~ save for later
	}

	const maxTry = 10 //~ undo at most this many bad steps
	const headRoom = 0.8
	const maxIterErr = 0.1
	const maxIter = 5
	const alpha_ref = 0.6

	try := 0
	restrict_step := false

	for {

		dt := engine.dt.Scalar()
		badStep := false
		badIterator := false

		er := make([]float64, len(equation))
		alp := make([]float64, len(equation))

		//~ Do zero-order approximation with explicit Euler
		for i := range equation {
			y := equation[i].LHS()
			dy := equation[i].RHS()
			dyMul := dy.multiplier
			t_step := dt * dyMul[0]
			gpu.Madd(y.Array(), s.y0buffer[i], s.dy0buffer[i], t_step)
			y.Invalidate()
		}

		s.iterations.SetScalar(s.iterations.Scalar() + 1)

		//~ Do predictor using implicit Euler

		e.time.SetScalar(t0 + dt)
		e.UpdateEqRHS()

		for i := range equation {
			s.dybuffer[i].CopyFromDevice(equation[i].RHS().Array())
		}

		for i := range equation {
			er[i] = maxIterErr
		}

		iter := 0
		err := 1.0
		α := 0.0
		for {
			for i := range equation {

				y := equation[i].LHS()
				dy := equation[i].RHS()
				COMP := dy.NComp()
				srCOMP := 1.0 / math.Sqrt(float64(COMP))

				h := dt * dy.multiplier[0]
				gpu.Madd(s.ybuffer[i], s.y0buffer[i], dy.Array(), h)

				tErr := 0.0
				for p := 0; p < COMP; p++ {
					diffy := float64(s.diff[i].MaxDiff(y.Array().Component(p), s.ybuffer[i].Component(p)))
					maxy := float64(s.diff[i].MaxAbs(s.ybuffer[i].Component(p)))
					tErr += math.Pow(diffy / (s.maxAbsErr[i].Scalar()+maxy * s.maxRelErr[i].Scalar()), 2.0)
				}
				tErr = srCOMP * math.Sqrt(tErr)

				α = tErr / er[i]
				alp[i] = α
				s.alpha[i] = α
				er[i] = tErr

				y.Array().CopyFromDevice(s.ybuffer[i])
				y.Invalidate()
			}

			//~ Get the largest error
			sort.Float64s(er)
			sort.Float64s(alp)
			err = er[len(equation)-1]
			α = alp[len(equation)-1]

			iter = iter + 1
			s.iterations.SetScalar(s.iterations.Scalar() + 1)
			//~ Check first if the target error is reached
			if err <= maxIterErr {
				break
			}
			//~ If not, then check for convergence
			if α >= 1.0 || iter > maxIter {
				badIterator = true
				break
			}
			e.UpdateEqRHS()
		}
		//~ If fixed-point iterator cannot converge, then panic
		if badIterator && try == (maxTry-1) {
			panic(Bug(fmt.Sprintf("The BDF Euler iterator cannot converge! Please increase the maximum number of iterations and re-run!")))
		} else if badIterator {
			//~ if there is a bad step in iterator then do hard/soft for step correction for fast/slow convergence
			h_alpha := 0.5 * dt
			if α > alpha_ref {
				h_alpha = dt * math.Pow(alpha_ref / α, 0.5)
			}
			engine.dt.SetScalar(h_alpha)
			restrict_step = true
			continue
		}

		//~ Save function value of the comparator
		//~ and restore dy as estimated by explicit Euler
		for i := range equation {
			s.y1buffer[i].CopyFromDevice(equation[i].LHS().Array())
			equation[i].RHS().Array().CopyFromDevice(s.dybuffer[i])
		}

		//~ Apply embedded 2nd order implicit method (trapezoidal)

		for i := range equation {
			er[i] = maxIterErr
		}

		iter = 0
		err = 1.0
		badIterator = false

		for {
			for i := range equation {

				y := equation[i].LHS()
				dy := equation[i].RHS()
				COMP := dy.NComp()
				srCOMP := 1.0 / math.Sqrt(float64(COMP))

				h := float64(dt * dy.multiplier[0])
				gpu.AddMadd(s.ybuffer[i], s.y0buffer[i], dy.Array(), s.dy0buffer[i], 0.5*h)

				tErr := 0.0
				for p := 0; p < COMP; p++ {
					diffy := float64(s.diff[i].MaxDiff(y.Array().Component(p), s.ybuffer[i].Component(p)))
					maxy := float64(s.diff[i].MaxAbs(s.ybuffer[i].Component(p)))
					tErr += math.Pow(diffy/(s.maxAbsErr[i].Scalar()+maxy*s.maxRelErr[i].Scalar()), 2.0)
				}
				tErr = srCOMP * math.Sqrt(tErr)

				alp[i] = tErr / er[i]
				er[i] = tErr

				y.Array().CopyFromDevice(s.ybuffer[i])
				y.Invalidate()
			}
			//~ Get the largest error
			sort.Float64s(er)
			sort.Float64s(alp)
			err = er[len(equation)-1]
			α = alp[len(equation)-1]

			iter = iter + 1
			s.iterations.SetScalar(s.iterations.Scalar() + 1)
			//~ Check first if the target error is reached
			if err <= maxIterErr {
				break
			}
			//~ If not, then check for convergence
			if α >= 1.0 || iter > maxIter {
				badIterator = true
				break
			}
			e.UpdateEqRHS()
		}

		if badIterator && try == (maxTry-1) {
			//~ If fixed-point iterator cannot converge, then panic
			panic(Bug(fmt.Sprintf("The BDF Trapezoidal iterator cannot converge! Please decrease the error the maximum number of iterations and re-run!")))
		} else if badIterator {
			//~ if there is a bad step in iterator then do hard/soft for step correction for fast/slow convergence
			h_alpha := 0.5 * dt
			if α > alpha_ref {
				h_alpha = dt * math.Pow(alpha_ref / α, 0.5)
			}
			engine.dt.SetScalar(h_alpha)
			continue
		}

		for i := range equation {

			y := equation[i].LHS()
			COMP := y.NComp()
			srCOMP := 1.0 / math.Sqrt(float64(COMP))

			tErr := 0.0
			for p := 0; p < COMP; p++ {
				diffy := float64(s.diff[i].MaxDiff(y.Array().Component(p), s.y1buffer[i].Component(p)))
				maxy := float64(s.diff[i].MaxAbs(s.y1buffer[i].Component(p)))
				tErr += math.Pow(diffy/(s.maxAbsErr[i].Scalar()+maxy*s.maxRelErr[i].Scalar()), 2.0)
			}
			tErr = srCOMP * math.Sqrt(tErr)

			if tErr > 1.0 {
				s.badSteps.SetScalar(s.badSteps.Scalar() + 1)
				badStep = true
			}
			s.err[i].SetScalar(tErr)
			//~ Estimate step correction
			step_corr := math.Pow(headRoom/tErr, 0.5)

			h_r := dt * step_corr
			new_dt := h_r
			//~ if iterator reported convergence problems, then the step correction should be restricted according to the linear prediction of the sweet convergence spot.
			if restrict_step {
				h_alpha := dt * math.Pow(alpha_ref / s.alpha[i], 0.5)
				new_dt = math.Min(h_r, h_alpha)
				restrict_step = false
			}

			//~ User-defined limiter for the new step. Just for stability experiments.
			if new_dt < s.minDt.Scalar() {
				new_dt = s.minDt.Scalar()
			}
			if new_dt > s.maxDt.Scalar() {
				new_dt = s.maxDt.Scalar()
			}

			s.newDt[i] = new_dt

			//~ Keep the history of 'good' errors
			if !badStep {
				s.err_list[i].PushFront(tErr)
				s.steps_list[i].PushFront(dt)

				if s.err_list[i].Len() == 10 {
					s.err_list[i].Remove(s.err_list[i].Back())
					s.steps_list[i].Remove(s.steps_list[i].Back())
				}
			}
		}
		//~ Get the new timestep
		sort.Float64s(s.newDt)
		nDt := s.newDt[0]
		engine.dt.SetScalar(nDt)
		if !badStep || nDt == s.minDt.Scalar() {
			break
		}
		if try > maxTry {
			panic(Bug(fmt.Sprint("The solver cannot converge after ", maxTry, " badsteps")))
		}

		try++
	}
	//~ Advance step
	e.step.SetScalar(e.step.Scalar() + 1) // advance time step
}

func (s *BDFAM12) Dependencies() (children, parents []string) {
	children = []string{"dt", "bdf_iterations", "t", "step", "badsteps"}
	parents = []string{"dt", "mindt", "maxdt"}
	for i := range s.err {
		parents = append(parents, s.maxAbsErr[i].Name())
		parents = append(parents, s.maxRelErr[i].Name())
	}
	return
}

// Register this module
func init() {
	RegisterModule("solver/am12", "Adaptive Adams-Moulton 1+2 solver", LoadBDFAM12)
}

func LoadBDFAM12(e *Engine) {
	s := new(BDFAM12)

	// Minimum/maximum time step
	s.minDt = e.AddNewQuant("mindt", SCALAR, VALUE, Unit("s"), "Minimum time step")
	s.minDt.SetScalar(1e-38)
	s.minDt.SetVerifier(Positive)
	s.maxDt = e.AddNewQuant("maxdt", SCALAR, VALUE, Unit("s"), "Maximum time step")
	s.maxDt.SetVerifier(Positive)
	s.maxDt.SetScalar(1e38)

	s.iterations = e.AddNewQuant("bdf_iterations", SCALAR, VALUE, Unit(""), "Number of iterations per step")
	s.badSteps = e.AddNewQuant("badsteps", SCALAR, VALUE, Unit(""), "Number of time steps that had to be re-done")

	equation := e.equation
	s.ybuffer = make([]*gpu.Array, len(equation))
	s.y0buffer = make([]*gpu.Array, len(equation))
	s.y1buffer = make([]*gpu.Array, len(equation))
	s.dy0buffer = make([]*gpu.Array, len(equation))
	s.dybuffer = make([]*gpu.Array, len(equation))

	s.err = make([]*Quant, len(equation))
	s.alpha = make([]float64, len(equation))
	s.err_list = make([]*list.List, len(equation))
	s.steps_list = make([]*list.List, len(equation))

	for i := range equation {
		s.err_list[i] = list.New()
		s.steps_list[i] = list.New()
	}

	s.maxAbsErr = make([]*Quant, len(equation))
	s.maxRelErr = make([]*Quant, len(equation))
	s.diff = make([]gpu.Reductor, len(equation))
	s.newDt = make([]float64, len(equation))

	for i := range equation {

		eqn := &(equation[i])
		Assert(eqn.kind == EQN_PDE1)
		out := eqn.LHS()
		unit := out.Unit()
		s.err[i] = e.AddNewQuant(out.Name()+"_error", SCALAR, VALUE, unit, "Error/step estimate for "+out.Name())
		s.maxAbsErr[i] = e.AddNewQuant(out.Name()+"_maxAbsError", SCALAR, VALUE, unit, "Maximum absolute error per step for "+out.Name())
		s.maxAbsErr[i].SetScalar(1e-4)
		s.maxRelErr[i] = e.AddNewQuant(out.Name()+"_maxRelError", SCALAR, VALUE, Unit(""), "Maximum relative error per step for "+out.Name())
		s.maxRelErr[i].SetScalar(1e-3)
		s.diff[i].Init(1, out.Array().Size3D())

		s.maxAbsErr[i].SetVerifier(Positive)
		s.maxRelErr[i].SetVerifier(Positive)

		// TODO: recycle?

		y := equation[i].LHS()
		s.ybuffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.y0buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.y1buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.dy0buffer[i] = Pool.Get(y.NComp(), y.Size3D())
		s.dybuffer[i] = Pool.Get(y.NComp(), y.Size3D())

	}
	e.SetSolver(s)
}
