//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// CGO wrappers for llbar-nonlocal00nc.h
// Author: Mykola Dvornik

//#include "libhotspin.h"
import "C"

import (
	. "hotspin-core/common"
	"unsafe"
)

func LLBarNonlocal00NC(t *Array, h *Array, msat0T0 *Array, lambda_e *Array, lambda_eMul []float64, cellsizeX float64, cellsizeY float64, cellsizeZ float64, pbc []int) {

	// Bookkeeping
	CheckSize(t.Size3D(), h.Size3D())
	Assert(h.NComp() == 3)

	// Calling the CUDA functions
	C.llbar_nonlocal00nc_async(
		(*C.double)(unsafe.Pointer(uintptr(t.Comp[X].pointer))),
		(*C.double)(unsafe.Pointer(uintptr(t.Comp[Y].pointer))),
		(*C.double)(unsafe.Pointer(uintptr(t.Comp[Z].pointer))),

		(*C.double)(unsafe.Pointer(uintptr(h.Comp[X].pointer))),
		(*C.double)(unsafe.Pointer(uintptr(h.Comp[Y].pointer))),
		(*C.double)(unsafe.Pointer(uintptr(h.Comp[Z].pointer))),

		(*C.double)(unsafe.Pointer(uintptr(msat0T0.Comp[X].pointer))),

		(*C.double)(unsafe.Pointer(uintptr(lambda_e.Comp[X].pointer))),
		(*C.double)(unsafe.Pointer(uintptr(lambda_e.Comp[Y].pointer))),
		(*C.double)(unsafe.Pointer(uintptr(lambda_e.Comp[Z].pointer))),

		(C.double)(float64(lambda_eMul[X])),
		(C.double)(float64(lambda_eMul[Y])),
		(C.double)(float64(lambda_eMul[Z])),

		(C.int)(t.PartSize()[X]),
		(C.int)(t.PartSize()[Y]),
		(C.int)(t.PartSize()[Z]),

		(C.double)(cellsizeX),
		(C.double)(cellsizeY),
		(C.double)(cellsizeZ),

		(C.int)(pbc[X]),
		(C.int)(pbc[Y]),
		(C.int)(pbc[Z]),

		(C.CUstream)(unsafe.Pointer(uintptr(t.Stream))))
}
