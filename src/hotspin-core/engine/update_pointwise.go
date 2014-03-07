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

// Updates a quantity according to a point-wise defined function of time.
type PointwiseUpdater struct {
	quant   *Quant
	lastIdx int         // Index of last time, for fast lookup of next
	points  [][]float64 // List of time+value lines: [time0, valx, valy, valz], [time1, ...
}

func newPointwiseUpdater(q *Quant) *PointwiseUpdater {
	u := new(PointwiseUpdater)
	u.quant = q
	u.points = make([][]float64, 0, 100)
	engine.Depends(q.Name(), "t") // declare time-dependence
	return u
}

func (field *PointwiseUpdater) Update() {

	//Debug(field)

	if len(field.points) < 2 {
		panic(InputErr("Pointwise definition needs at least two points"))
	}
	time := engine.time.Scalar()

	//find closest times

	// first search backwards in time,
	// multi-stage solvers may have gone back in time.
	i := field.lastIdx
	if i >= len(field.points) {
		i = len(field.points) - 1
	}
	for ; i > 0; i-- {
		if field.points[i][0] <= time {
			break
		}
	}

	if i < 0 {
		i = 0
	}

	// then search forward
	for ; i < len(field.points); i++ {
		if field.points[i][0] >= time {
			break
		}
	}

	// i now points to a time >= engine.time
	field.lastIdx = i

	if i >= len(field.points) {
		panic(InputErrF("Out of range of pointwise-defined quantity", field.quant.Name(), ". Field is defined only up to t=", field.points[len(field.points)-1][0], "s, but requested at t=", time, "s. Please define the quantity up to larger t."))
	}

	value := field.quant.multiplier
	// out of range: value = 0
	if i-1 < 0 || i < 0 {
		for i := range value {
			value[i] = 0
		}
		return
	}

	t1 := field.points[i-1][0]
	t2 := field.points[i][0]
	v1 := field.points[i-1][1:]
	v2 := field.points[i][1:]
	dt := t2 - t1         //pt2[0] - pt1[0]
	t := (time - t1) / dt // 0..1

	Assert(t >= 0 && t <= 1)
	for i := range value {
		value[i] = v1[i] + t*(v2[i]-v1[i])
	}
	field.quant.Invalidate() //SetValue(value) //?
	//Debug("pointwise update", field.quant.Name(), "time=", time, "i=", i, "value=", value)
}

func (p *PointwiseUpdater) Append(time float64, value []float64) {
	nComp := p.quant.NComp()
	checkComp(p.quant, len(value))
	if len(p.points) > 0 {
		if p.points[len(p.points)-1][0] > time {
			Debug(InputErrF("Pointwise definition should be in chronological order, but", p.points[len(p.points)-1][0], ">", time, ". Please skip this warning if it is desired behaviour. The list of points will be flushed."))
			p.points = p.points[len(p.points):]
		}
	}

	entry := make([]float64, nComp+1)
	entry[0] = time
	copy(entry[1:], value)
	p.points = append(p.points, entry)

}
