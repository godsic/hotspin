//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Author: Arne Vansteenkiste

import (
	"fmt"
	"io"
	. "hotspin-core/common"
)

// Table refers to an open data table 
// to which space-independent output is appended during the simulation.
type Table struct {
	out   io.WriteCloser
	fname string
}

// New table that will write in the file.
func NewTable(fname string) *Table {
	t := new(Table)
	t.fname = fname
	return t
}

// Append the quantities value to the table.
func (t *Table) Tabulate(quants []string) {
	if t.out == nil {
		t.out = OpenWRONLY(t.fname)
		writeTableHeader(t.out, quants)
	}
	e := GetEngine()
	for _, q := range quants {
		quant := e.Quant(q)
		quant.Update() //!!
		v := quant.multiplier
		n := len(v)
		for i := n - 1; i >= 0; i-- { // Swap XYZ -> ZYX
			fmt.Fprint(t.out, v[i], "\t")
		}
	}
	fmt.Fprintln(t.out)
}

func (t *Table) Close() {
	t.out.Close()
}

func writeTableHeader(out io.Writer, quants []string) {
	e := GetEngine()
	fmt.Fprint(out, "#")
	for _, q := range quants {
		quant := e.Quant(q)
		checkKinds(quant, VALUE, MASK)
		n := quant.NComp()
		for i := n - 1; i >= 0; i-- {
			comp := ""
			if n > 1 {
				comp = "_" + string('z'-i)
			}
			fmt.Fprint(out, quant.Name()+comp, " (", quant.Unit(), ")\t")
		}
	}
	fmt.Fprintln(out)
}
