//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// CGO wrappers for baryakhtar-torque.cu
// Author: Mykola Dvornik

//#include "libhotspin.h"
import "C"

import (
	. "hotspin-core/common"
	"unsafe"
)

func LLBarTorqueAsync(t *Array, M *Array, h *Array, msat0T0 *Array) {

	// Bookkeeping
	CheckSize(h.Size3D(), M.Size3D())

	Assert(h.NComp() == 3)

	C.llbar_torque_async(
		(*C.double)(unsafe.Pointer(uintptr(t.Comp[X].pointer))),
		(*C.double)(unsafe.Pointer(uintptr(t.Comp[Y].pointer))),
		(*C.double)(unsafe.Pointer(uintptr(t.Comp[Z].pointer))),

		(*C.double)(unsafe.Pointer(uintptr(M.Comp[X].pointer))),
		(*C.double)(unsafe.Pointer(uintptr(M.Comp[Y].pointer))),
		(*C.double)(unsafe.Pointer(uintptr(M.Comp[Z].pointer))),

		(*C.double)(unsafe.Pointer(uintptr(h.Comp[X].pointer))),
		(*C.double)(unsafe.Pointer(uintptr(h.Comp[Y].pointer))),
		(*C.double)(unsafe.Pointer(uintptr(h.Comp[Z].pointer))),

		(*C.double)(unsafe.Pointer(uintptr(msat0T0.Comp[X].pointer))),

		(C.CUstream)(unsafe.Pointer(uintptr(t.Stream))),
		(C.int)(t.partLen3D))
}
