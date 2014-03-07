//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// Authors: Mykola Dvornik
// has no multi-gpu support

import (
	"cuda/cufft"
	. "hotspin-core/common"
)

//Register this FFT plan
func init() {
	fftPlans["X"] = NewFFTPlanX
}

type FFTPlanX struct {
	//sizes
	dataSize   [3]int // Size of the (non-zero) input data block
	logicSize  [3]int // Transform size including zero-padding. >= dataSize
	outputSize [3]int // Size of the output data (one extra row PER GPU)
	buffer     Array  // An array for zero-padding the data

	// fft plans
	plan3dR2C cufft.Handle
	plan3dC2R cufft.Handle
	Stream
}

func (fft *FFTPlanX) init(dataSize, logicSize []int) {

	Assert(len(dataSize) == 3)
	Assert(len(logicSize) == 3)
	const nComp = 1

	fft.buffer.Init(nComp, logicSize, DONT_ALLOC)

	Debug(dataSize, logicSize)

	outputSize := FFTOutputSize(logicSize)
	for i := range fft.dataSize {
		fft.dataSize[i] = dataSize[i]
		fft.logicSize[i] = logicSize[i]
		fft.outputSize[i] = outputSize[i]
	}

	fft.Stream = NewStream()

	fft.plan3dR2C = cufft.Plan3d(fft.logicSize[0], fft.logicSize[1], fft.logicSize[2], cufft.R2C)
	fft.plan3dR2C.SetStream(uintptr(fft.Stream))
	fft.plan3dR2C.SetCompatibilityMode(cufft.COMPATIBILITY_NATIVE)
	fft.plan3dC2R = cufft.Plan3d(fft.logicSize[0], fft.logicSize[1], fft.logicSize[2], cufft.C2R)
	fft.plan3dC2R.SetStream(uintptr(fft.Stream))
	fft.plan3dC2R.SetCompatibilityMode(cufft.COMPATIBILITY_NATIVE)
}

func NewFFTPlanX(dataSize, logicSize []int) FFTInterface {
	fft := new(FFTPlanX)
	fft.init(dataSize, logicSize)
	return fft
}

func (fft *FFTPlanX) Free() {
	for i := range fft.dataSize {
		fft.dataSize[i] = 0
		fft.logicSize[i] = 0
	}
}

func (fft *FFTPlanX) Forward(in, out *Array) {
	AssertMsg(in.size4D[0] == 1, "1")
	AssertMsg(out.size4D[0] == 1, "2")
	CheckSize(in.size3D, fft.dataSize[:])
	CheckSize(out.size3D, fft.outputSize[:])

	buf := &fft.buffer
	buf.PointTo(out, 0)

	buf.Zero()
	CopyPad3D(buf, in)

	ptr := uintptr(out.pointer)

	fft.plan3dR2C.ExecR2C(ptr, ptr)
	fft.Sync()
}

func (fft *FFTPlanX) Inverse(in, out *Array) {

	ptr := uintptr(in.pointer)
	fft.plan3dC2R.ExecC2R(ptr, ptr)
	fft.Sync()

	buf := &fft.buffer
	buf.PointTo(in, 0)

	CopyUnPad3D(out, buf)
}
