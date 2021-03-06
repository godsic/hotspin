//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// CGO wrappers for llbar-local00nc.h
// Author: Mykola Dvornik

//#include "libhotspin.h"
import "C"

import (
	. "hotspin-core/common"
	"unsafe"
)

func LLBarLocal00NC(t *Array, h *Array, msat0T0 *Array, lambda *Array, lambdaMul []float64) {

	// Bookkeeping
	CheckSize(h.Size3D(), t.Size3D())
	Assert(h.NComp() == 3)

	// Calling the CUDA functions
	C.llbar_local00nc_async(
		(*C.double)(unsafe.Pointer(uintptr(t.Comp[X].pointer))),
		(*C.double)(unsafe.Pointer(uintptr(t.Comp[Y].pointer))),
		(*C.double)(unsafe.Pointer(uintptr(t.Comp[Z].pointer))),

		(*C.double)(unsafe.Pointer(uintptr(h.Comp[X].pointer))),
		(*C.double)(unsafe.Pointer(uintptr(h.Comp[Y].pointer))),
		(*C.double)(unsafe.Pointer(uintptr(h.Comp[Z].pointer))),

		(*C.double)(unsafe.Pointer(uintptr(msat0T0.Comp[X].pointer))),

		(*C.double)(unsafe.Pointer(uintptr(lambda.Comp[X].pointer))), //XX
		(*C.double)(unsafe.Pointer(uintptr(lambda.Comp[Y].pointer))), //YY
		(*C.double)(unsafe.Pointer(uintptr(lambda.Comp[Z].pointer))), //ZZ

		(C.double)(float64(lambdaMul[X])),
		(C.double)(float64(lambdaMul[Y])),
		(C.double)(float64(lambdaMul[Z])),

		(C.CUstream)(unsafe.Pointer(uintptr(t.Stream))),

		(C.int)(t.partLen3D))
}
