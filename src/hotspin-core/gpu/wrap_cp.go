//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// CGO wrappers for cp.cu
// Author: Mykola Dvornik

//#include "libhotspin.h"
import "C"

import (
	. "hotspin-core/common"
	"unsafe"
)

func CpAsync(cp *Array, T *Array, Td *Array, n *Array, TdMul float64, stream Stream) {

	// Bookkeeping
	CheckSize(cp.Size3D(), T.Size3D())

	// Calling the CUDA functions
	C.cpAsync(
		(*C.double)(unsafe.Pointer(uintptr(cp.Comp[X].pointer))),
		(*C.double)(unsafe.Pointer(uintptr(T.Comp[X].pointer))),
		(*C.double)(unsafe.Pointer(uintptr(Td.Comp[X].pointer))),
		(*C.double)(unsafe.Pointer(uintptr(n.Comp[X].pointer))),

		(C.double)(TdMul),

		(C.int)(cp.partLen3D),
		(C.CUstream)(unsafe.Pointer(uintptr(stream))))
}
