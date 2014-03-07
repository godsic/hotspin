//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// This file implements generation of a graphviz .dot file
// representing the physics graph.
// Author: Arne Vansteenkiste

import (
	"fmt"
	"io"
	. "hotspin-core/common"
	"os/exec"
	"strings"
)

// Write .dot file for graphviz, 
// representing the physics graph.
func (e *Engine) WriteDot(out io.Writer) {
	fmt.Fprintln(out, "digraph Physics{")
	fmt.Fprintln(out, "rankdir=LR")

	// Add quantities
	quants := e.quantity
	for _, v := range quants {
		k := sanitize(v.Name())
		label := "label=" + `"` + v.FullName()
		//if v.desc != "" {
		//	label += "\\n(" + v.desc + `)"`
		//} else {
		label += `"`
		//}
		fmt.Fprintln(out, k, " [shape=box, group=", k[0:1], label, "];") // use first letter as group name.
		// Add dependencies
		for _, c := range v.children {
			fmt.Fprintln(out, k, "->", sanitize(c.name), ";")
		}
	}

	//Add solver cluster node
	if len(e.equation) > 0 {
		fmt.Fprintln(out, "subgraph cluster0{")
		fmt.Fprintln(out, "rank=sink;")
		for i, _ := range e.equation {
			ODE := "solver" + fmt.Sprint(i)
			fmt.Fprintln(out, ODE+` [style=filled, shape=box, label="`, e.equation[i].String(), `"];`)
		}
		fmt.Fprintln(out, "}")

		// Special dependencies of solver
		if e.solver != nil {
			children, parents := e.solver.Dependencies()
			for _, c := range children {
				fmt.Fprintln(out, "subgraph cluster0 ->", c, ";")
			}
			for _, p := range parents {
				fmt.Fprintln(out, p, "-> subgraph cluster0;")
			}
		}
		fmt.Fprintln(out, "{rank=sink;", "subgraph cluster0", "};")
	}

	// Add ODE node
	for i, eqn := range e.equation {
		ODE := "solver" + fmt.Sprint(i)
		inp, outp := eqn.input, eqn.output
		for j := range outp {
			fmt.Fprintln(out, ODE, "->", outp[j].Name(), ";")
			fmt.Fprintln(out, "{rank=source;", outp[j].Name(), "};")
			fmt.Fprintln(out, "{rank=sink;", inp[j].Name(), "};")
		}
		for j := range inp {
			fmt.Fprintln(out, inp[j].Name(), "->", ODE, ";")
		}
	}

	// align similar nodes
	i := 0
	for _, a := range quants {
		j := 0
		for _, b := range quants {
			if i < j {
				if similar(a.Name(), b.Name()) {
					fmt.Fprintln(out, "{rank=same;", a.Name(), ";", b.Name(), "};")
				}
			}
			j++
		}
		i++
	}

	fmt.Fprintln(out, "}")
}

// replaces characters that graphviz cannot handle as labels.
func sanitize(in string) (out string) {
	out = strings.Replace(in, "<", "_leftavg_", -1)
	out = strings.Replace(out, ">", "_rightavg_", -1)
	out = strings.Replace(out, ".", "_dot_", -1)
	out = strings.Replace(out, "~", "_tilda_", -1)
	return
}

// Executes dot -Tformat -O infile
// rendering the dot input file.
func RunDot(infile, format string) {
	dot, err := exec.LookPath("dot")
	if err != nil {
		Warn("could not find dot in PATH", err)
		return
	}
	Debug("exec", dot, "-T"+format, "-O", infile)
	proc := exec.Command(dot, "-T"+format, "-O", infile)
	out, err2 := proc.CombinedOutput()
	if err2 != nil {
		Warn(dot, infile, "failed:", err2, string(out))
	}
}

// true if a and b are similar names, to be equally ranked in the dot graph.
func similar(a, b string) (similar bool) {
	defer func() {
		if recover() != nil {
			return
		}
	}()
	similar = a[0] == b[0] && a[1] == '_' && b[1] == '_'
	return
}
