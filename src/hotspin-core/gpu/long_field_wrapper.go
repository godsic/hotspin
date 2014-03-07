//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// CGO wrappers for long_field.cu
// Author: Mykola Dvornik, Arne Vansteenkiste

//#include "libhotspin.h"
import "C"

import (
	. "hotspin-core/common"
	"unsafe"
)

func LongFieldAsync(hlf *Array, m *Array, msat0T0 *Array, J *Array, n *Array, Tc *Array, Ts *Array, msat0T0Mul float64, JMul float64, nMul float64, TcMul float64, TsMul float64, stream Stream) {

	// Bookkeeping
	CheckSize(hlf.Size3D(), m.Size3D())

	//CheckSize(msat0.Size3D(), m.Size3D()) // since it could be any!

	//Assert(msat0.NComp() == 1) no clue how it shoud work if mask is not there

	// Calling the CUDA functions
	C.long_field_async(
		(*C.float)(unsafe.Pointer((hlf.Comp[X].pointer))),
		(*C.float)(unsafe.Pointer((hlf.Comp[Y].pointer))),
		(*C.float)(unsafe.Pointer((hlf.Comp[Z].pointer))),

		(*C.float)(unsafe.Pointer((m.Comp[X].pointer))),
		(*C.float)(unsafe.Pointer((m.Comp[Y].pointer))),
		(*C.float)(unsafe.Pointer((m.Comp[Z].pointer))),

		(*C.float)(unsafe.Pointer((msat0T0.Comp[X].pointer))),
		(*C.float)(unsafe.Pointer((J.Comp[X].pointer))),
		(*C.float)(unsafe.Pointer((n.Comp[X].pointer))),

		(*C.float)(unsafe.Pointer((Tc.Comp[X].pointer))),
		(*C.float)(unsafe.Pointer((Ts.Comp[X].pointer))),

		(C.float)(msat0T0Mul),
		(C.float)(JMul),
		(C.float)(nMul),

		(C.float)(TcMul),
		(C.float)(TsMul),

		(C.int)(hlf.partLen3D),
		(C.CUstream)(unsafe.Pointer((stream))))
}
