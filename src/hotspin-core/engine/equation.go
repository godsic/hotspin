//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Author: Arne Vansteenkiste

// Type representing a differential equation
type Equation struct {
	input, output []*Quant // input/output quantities
	kind          int      // type of equation
}

// d output / d t = input
func PDE1(output, input *Quant) Equation {
	return Equation{[]*Quant{input}, []*Quant{output}, EQN_PDE1}
}

func (e *Equation) String() string {
	switch e.kind {
	case EQN_PDE1:
		return "∂" + e.output[0].Name() + "/∂t=" + e.input[0].Name()
	}
	return "<invalid equation>"
}

func (e *Equation) UpdateRHS(){
	e.input[0].Update()
}

func (e *Equation) LHS() *Quant {
	return e.output[0]
}

func (e *Equation) RHS() *Quant {
	return e.input[0]
}

const (
	EQN_INVALID = iota // not used
	EQN_PDE1           // dy/dt=f(y,t)
)
