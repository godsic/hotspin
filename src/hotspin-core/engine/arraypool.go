//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Author: Arne Vansteenkiste

import (
	"hotspin-core/gpu"
)

// the global array pool
var Pool ArrayPool

type ArrayPool struct {
}

func (p *ArrayPool) Get(nComp int, size []int) *gpu.Array {
	// TODO: actual recycling
	return gpu.NewArray(nComp, size)
}

func (p *ArrayPool) Recycle(array **gpu.Array) {
	// TODO: actual recycling
	(*array).Free()
	(*array) = nil

}
