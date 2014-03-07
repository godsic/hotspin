//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// Author: Arne Vansteenkiste

import (
	cu "cuda/driver"
	"flag"
	. "hotspin-core/common"
)

//const BIG = 16 * 1024 * 1024
const BIG = 16 * 128 * 512

func init() {
	flag.Parse()
	InitLogger("test.log")
	cu.Init()
	//InitAllGPUs(0)
	//println("		*****  u s i n g    1    g p u  *******  ")
	//InitMultiGPU([]int{0}, 0)
	InitDebugGPUs()
}

// return a few array sizes for testing
func sizes() [][]int {
	return [][]int{{2, 2 * NDevice(), 8}, {32, 8 * NDevice(), 4}}
}

func bigsize() []int {
	return []int{8, 256 * NDevice(), 1024}
}

// return a few numbers of components for testing
func comps() []int {
	return []int{1, 3}
}
