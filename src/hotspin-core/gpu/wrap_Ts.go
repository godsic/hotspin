//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// CGO wrappers for langevin.cu
// Author: Mykola Dvornik

//#include "libhotspin.h"
import "C"

import (
	. "hotspin-core/common"
	"unsafe"
)

func TsSync(Ts *Array, msat *Array, msat0T0 *Array, Tc *Array, S *Array, msatMul float64, msat0T0Mul float64, TcMul float64, SMul float64) {

	// Bookkeeping
	CheckSize(msat.Size3D(), Ts.Size3D())

	// Calling the CUDA functions
	C.tsAsync(
		(*C.double)(unsafe.Pointer(uintptr(Ts.Comp[X].pointer))),
		(*C.double)(unsafe.Pointer(uintptr(msat.Comp[X].pointer))),
		(*C.double)(unsafe.Pointer(uintptr(msat0T0.Comp[X].pointer))),
		(*C.double)(unsafe.Pointer(uintptr(Tc.Comp[X].pointer))),
		(*C.double)(unsafe.Pointer(uintptr(S.Comp[X].pointer))),

		(C.double)(msatMul),
		(C.double)(msat0T0Mul),
		(C.double)(TcMul),
		(C.double)(SMul),

		(C.int)(Ts.partLen3D),
		(C.CUstream)(unsafe.Pointer(uintptr(Ts.Stream))))
	Ts.Stream.Sync()
}
