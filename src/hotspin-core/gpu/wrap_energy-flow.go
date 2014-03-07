//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// CGO wrappers for kappa.cu
// Author: Mykola Dvornik

//#include "libhotspin.h"
import "C"

import (
	. "hotspin-core/common"
	"unsafe"
)

func EnergyFlowAsync(w *Array, mf *Array, R *Array, Tc *Array, S *Array, n *Array, SMul float64, stream Stream) {

	// Calling the CUDA functions
	C.energyFlowAsync(
		(*C.float)(unsafe.Pointer(uintptr(w.Comp[X].pointer))),

		(*C.float)(unsafe.Pointer(uintptr(mf.Comp[X].pointer))),
		(*C.float)(unsafe.Pointer(uintptr(mf.Comp[Y].pointer))),
		(*C.float)(unsafe.Pointer(uintptr(mf.Comp[Z].pointer))),

		(*C.float)(unsafe.Pointer(uintptr(R.Comp[X].pointer))),
		(*C.float)(unsafe.Pointer(uintptr(R.Comp[Y].pointer))),
		(*C.float)(unsafe.Pointer(uintptr(R.Comp[Z].pointer))),

		(*C.float)(unsafe.Pointer(uintptr(Tc.Comp[X].pointer))),
		(*C.float)(unsafe.Pointer(uintptr(S.Comp[X].pointer))),
		(*C.float)(unsafe.Pointer(uintptr(n.Comp[X].pointer))),

		(C.float)(SMul),

		(C.int)(w.partLen3D),
		(C.CUstream)(unsafe.Pointer(uintptr(stream))))
}
