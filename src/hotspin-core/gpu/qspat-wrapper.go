//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// CGO wrappers for Qspat.cu
// Author: Mykola Dvornik

//#include "libhotspin.h"
import "C"

import (
	. "hotspin-core/common"
	"unsafe"
)

func Qspat_async(Q *Array, T *Array, k *Array, kMul []float64, cs []float64, pbc []int) {

	// Calling the CUDA functions
	C.Qspat_async(
		(*C.float)(unsafe.Pointer((Q.Comp[X].pointer))),
		(*C.float)(unsafe.Pointer((T.Comp[X].pointer))),
		(*C.float)(unsafe.Pointer((k.Comp[X].pointer))),

		(C.float)(float32(kMul[0])),

		(C.int)(Q.PartSize()[X]),
		(C.int)(Q.PartSize()[Y]),
		(C.int)(Q.PartSize()[Z]),

		(C.float)(float32(cs[X])),
		(C.float)(float32(cs[Y])),
		(C.float)(float32(cs[Z])),

		(C.int)(pbc[X]),
		(C.int)(pbc[Y]),
		(C.int)(pbc[Z]),
		(C.CUstream)(unsafe.Pointer((Q.Stream))))
}
