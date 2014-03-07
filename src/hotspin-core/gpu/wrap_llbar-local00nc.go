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
		(*C.float)(unsafe.Pointer((t.Comp[X].pointer))),
		(*C.float)(unsafe.Pointer((t.Comp[Y].pointer))),
		(*C.float)(unsafe.Pointer((t.Comp[Z].pointer))),

		(*C.float)(unsafe.Pointer((h.Comp[X].pointer))),
		(*C.float)(unsafe.Pointer((h.Comp[Y].pointer))),
		(*C.float)(unsafe.Pointer((h.Comp[Z].pointer))),

		(*C.float)(unsafe.Pointer((msat0T0.Comp[X].pointer))),

		(*C.float)(unsafe.Pointer((lambda.Comp[X].pointer))), //XX
		(*C.float)(unsafe.Pointer((lambda.Comp[Y].pointer))), //YY
		(*C.float)(unsafe.Pointer((lambda.Comp[Z].pointer))), //ZZ

		(C.float)(float32(lambdaMul[X])),
		(C.float)(float32(lambdaMul[Y])),
		(C.float)(float32(lambdaMul[Z])),

		(C.CUstream)(unsafe.Pointer((t.Stream))),

		(C.int)(t.partLen3D))
}
