//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// CGO wrappers for llbar_local02c.cu
// Author: Mykola Dvornik

//#include "libhotspin.h"
import "C"

import (
	. "hotspin-core/common"
	"unsafe"
)

func LLBarLocal02C(t *Array, m *Array, h *Array, msat0T0 *Array, mu *Array, muMul []float64) {

	// Bookkeeping
	CheckSize(h.Size3D(), m.Size3D())
	CheckSize(h.Size3D(), t.Size3D())
	Assert(h.NComp() == 3)

	// Calling the CUDA functions
	C.llbar_local02c_async(
		(*C.double)(unsafe.Pointer(uintptr(t.Comp[X].pointer))),
		(*C.double)(unsafe.Pointer(uintptr(t.Comp[Y].pointer))),
		(*C.double)(unsafe.Pointer(uintptr(t.Comp[Z].pointer))),

		(*C.double)(unsafe.Pointer(uintptr(m.Comp[X].pointer))),
		(*C.double)(unsafe.Pointer(uintptr(m.Comp[Y].pointer))),
		(*C.double)(unsafe.Pointer(uintptr(m.Comp[Z].pointer))),

		(*C.double)(unsafe.Pointer(uintptr(h.Comp[X].pointer))),
		(*C.double)(unsafe.Pointer(uintptr(h.Comp[Y].pointer))),
		(*C.double)(unsafe.Pointer(uintptr(h.Comp[Z].pointer))),

		(*C.double)(unsafe.Pointer(uintptr(msat0T0.Comp[X].pointer))),

		(*C.double)(unsafe.Pointer(uintptr(mu.Comp[X].pointer))), //XX
		(*C.double)(unsafe.Pointer(uintptr(mu.Comp[Y].pointer))), //YY
		(*C.double)(unsafe.Pointer(uintptr(mu.Comp[Z].pointer))), //ZZ

		(C.double)(float64(muMul[X])), //XX
		(C.double)(float64(muMul[Y])), //YY
		(C.double)(float64(muMul[Z])), //ZZ

		(C.CUstream)(unsafe.Pointer(uintptr(t.Stream))),
		(C.int)(t.partLen3D))
}
