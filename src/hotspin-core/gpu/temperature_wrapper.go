//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// CGO wrappers for temperature.cu
// Author: Arne Vansteenkiste

//#include "libhotspin.h"
import "C"
import (
	. "hotspin-core/common"
	"unsafe"
)

func ScaleNoiseAniz(h, mu, T, msat0T0 *Array,
	muMul []float64,
	KB2tempMul_mu0VgammaDtMsatMul float64) {
	CheckSize(h.Size3D(), mu.Size3D())
	C.temperature_scaleAnizNoise(
		(*C.float)(unsafe.Pointer((h.Comp[X].pointer))),
		(*C.float)(unsafe.Pointer((h.Comp[Y].pointer))),
		(*C.float)(unsafe.Pointer((h.Comp[Z].pointer))),

		(*C.float)(unsafe.Pointer((mu.Comp[X].pointer))), //XX
		(*C.float)(unsafe.Pointer((mu.Comp[Y].pointer))), //YY
		(*C.float)(unsafe.Pointer((mu.Comp[Z].pointer))), //ZZ

		(*C.float)(unsafe.Pointer((T.pointer))),
		(*C.float)(unsafe.Pointer((msat0T0.pointer))),

		(C.float)(float32(muMul[X])),
		(C.float)(float32(muMul[Y])),
		(C.float)(float32(muMul[Z])),

		(C.float)(float32(KB2tempMul_mu0VgammaDtMsatMul)),
		(C.CUstream)(unsafe.Pointer((h.Stream))),
		(C.int)(h.PartLen3D()))
}
