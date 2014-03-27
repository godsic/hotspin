//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

import (
	. "hotspin-core/common"
)

// Updates a quantity according to a point-wise defined function of argument.
type PointwiseOfUpdater struct {
	argument *Quant      // The argument
	quant    *Quant      // The function
	lastIdx  int         // Index of last time, for fast lookup of next
	points   [][]float64 // List of time+value lines: [arg0, valx, valy, valz], [arg1, ...
}

func newPointwiseOfUpdater(arg *Quant, q *Quant) *PointwiseOfUpdater {
	u := new(PointwiseOfUpdater)
	u.quant = q
	u.argument = arg
	u.points = make([][]float64, 0, 10000)
	engine.Depends(q.Name(), u.argument.Name()) // declare arg-dependence
	return u
}

func (field *PointwiseOfUpdater) Update() {

	//Debug(field)
	name := field.argument.Name()

	if len(field.points) < 2 {
		panic(InputErr("Pointwise definition needs at least two points"))
	}
	arg := field.argument.Scalar()

	//find closest arguments

	// first search backwards,
	// multi-stage solvers may have gone back

	i := field.lastIdx
	if i >= len(field.points) {
		i = len(field.points) - 1
	}
	for ; i > 0; i-- {
		if field.points[i][0] <= arg {
			break
		}
	}

	if i < 0 {
		i = 0
	}

	// then search forward
	for ; i < len(field.points); i++ {
		if field.points[i][0] >= arg {
			break
		}
	}

	// i now points to a arg >= engine.arg
	field.lastIdx = i

	if i >= len(field.points) {
		panic(InputErrF("Out of range of pointwise-defined quantity", field.quant.Name(), ". Field is defined only up to "+name+"= ", field.points[len(field.points)-1][0], field.quant.Unit(), ", but requested at"+name+"= ", arg, field.quant.Unit(), ". Please define the quantity up to larger "+name))
	}

	value := field.quant.multiplier
	// out of range: value = 0
	if i-1 < 0 || i < 0 {
		for i := range value {
			value[i] = 0
		}
		return
	}

	arg1 := field.points[i-1][0]
	arg2 := field.points[i][0]
	v1 := field.points[i-1][1:]
	v2 := field.points[i][1:]
	darg := arg2 - arg1          //pt2[0] - pt1[0]
	argum := (arg - arg1) / darg // 0..1
	//	Debug("arg:",arg, "arg1:",arg1, "arg2", arg2, "argum", argum)
	Assert(argum >= 0 && argum <= 1)
	for i := range value {
		value[i] = v1[i] + argum*(v2[i]-v1[i])
	}
	field.quant.Invalidate() //SetValue(value) //?
}

func (p *PointwiseOfUpdater) Append(arg float64, value []float64) {
	nComp := p.quant.NComp()
	checkComp(p.quant, len(value))
	if len(p.points) > 0 {
		if p.points[len(p.points)-1][0] > arg {
			panic(InputErrF("Pointwise definition should be in acsending order, but", p.points[len(p.points)-1][0], ">", arg))
		}
	}

	entry := make([]float64, nComp+1)
	entry[0] = arg
	copy(entry[1:], value)
	p.points = append(p.points, entry)

}

func (p *PointwiseOfUpdater) AppendMap(x []float64, y [][]float64) {
	nComp := p.quant.NComp()
	for i, _ := range x {
		checkComp(p.quant, len(y[i]))
	}
	for i := 0; i < len(x)-1; i++ {
		if x[i] >= x[i+1] {
			panic(InputErrF("Pointwise definition should be in acsending order, but", x[i], ">", x[i+1]))
		}
	}
	if len(p.points) > 0 {
		if p.points[len(p.points)-1][0] > x[0] {
			panic(InputErrF("When append, pointwise definition should be in acsending order, but", p.points[len(p.points)-1][0], ">", x[0]))
		}
	}

	for i, _ := range x {
		entry := make([]float64, nComp+1)
		entry[0] = x[i]
		checkComp(p.quant, len(y[i]))
		SwapXYZ(y[i])
		copy(entry[1:], y[i])
		p.points = append(p.points, entry)
	}
}
