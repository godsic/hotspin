//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Auhtor: Arne Vansteenkiste

import ()

// Saves a field (scalar field, vector field, etc) periodically.
type AutoSave struct {
	quant   string   // What to save. E.g. "m" for magnetization
	format  string   // Format to save in
	options []string // Output format options
	period  float64  // How often to save
	start   float64  // Starting point
	count   int      // Number of times it has been saved
}

// Called by the eninge
func (a *AutoSave) Notify(e *Engine) {
	t := e.time.Scalar() - a.start
	if t-float64(a.count)*a.period >= a.period {
		e.SaveAs(e.Quant(a.quant), a.format, a.options, e.AutoFilename(a.quant, a.format))
		a.count++
	}
}
