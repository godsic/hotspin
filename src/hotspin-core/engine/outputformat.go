//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Auhtor: Arne Vansteenkiste

import (
	"io"
	. "hotspin-core/common"
)

// A general output format for space-dependent quantities.
type OutputFormat interface {
	Name() string                                    // Name to register the format under. E.g. "txt". Also used as file extension
	Write(out io.Writer, q *Quant, options []string) // Writes the quantity buffer to out
}

// global map of registered output formats
var outputformats map[string]OutputFormat

// registers an output format
func RegisterOutputFormat(format OutputFormat) {
	if outputformats == nil {
		outputformats = make(map[string]OutputFormat)
	}
	outputformats[format.Name()] = format
}

// Retrieves an output format from its name. E.g. "txt", "omf"
func GetOutputFormat(name string) OutputFormat {
	f, ok := outputformats[name]
	if !ok {
		options := ""
		for k, _ := range outputformats {
			options += k + " "
		}
		panic(IOErr("Unknown output format: " + name + ". Options are: " + options))
	}
	return f
}
