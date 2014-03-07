//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// Author: Arne Vansteenkiste

import (
	. "hotspin-core/common"
)

func init() {
	fftPlans = make(map[string]func(d, l []int) FFTInterface)
}

// The default FFT constructor.
// The function pointer may be changed
// to use a different FFT implementation globally.
var NewDefaultFFT func(dataSize, logicSize []int) FFTInterface = NewFFTPlanX // this default is for tests, not sims.

// Global map with all registered FFT plans
var fftPlans map[string]func(dataSize, logicSize []int) FFTInterface

// Sets a global default FFT
func SetDefaultFFT(name string) {
	f, ok := fftPlans[name]
	if !ok {
		panic(InputErrF("Undefined FFT type:", name, "Options are", fftPlans))
	}
	NewDefaultFFT = f
}

// Interface for any sparse FFT plan.
type FFTInterface interface {
	Forward(in, out *Array)
	Inverse(in, out *Array)
	Free()
}

// Returns the normalization factor of an FFT with this logic size.
// (just the product of the sizes)
func FFTNormLogic(logicSize []int) int {
	return (logicSize[0] * logicSize[1] * logicSize[2])
}

// Returns the output size of an FFT with given logic size.
func FFTOutputSize(logicSize []int) []int {

	outputSize := make([]int, 3)
	outputSize[0] = logicSize[0]
	outputSize[1] = logicSize[1]
	outputSize[2] = logicSize[2] + 2

	return outputSize
}
