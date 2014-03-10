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
		(*C.double)(unsafe.Pointer(uintptr(h.Comp[X].pointer))),
		(*C.double)(unsafe.Pointer(uintptr(h.Comp[Y].pointer))),
		(*C.double)(unsafe.Pointer(uintptr(h.Comp[Z].pointer))),

		(*C.double)(unsafe.Pointer(uintptr(mu.Comp[X].pointer))), //XX
		(*C.double)(unsafe.Pointer(uintptr(mu.Comp[Y].pointer))), //YY
		(*C.double)(unsafe.Pointer(uintptr(mu.Comp[Z].pointer))), //ZZ

		(*C.double)(unsafe.Pointer(uintptr(T.pointer))),
		(*C.double)(unsafe.Pointer(uintptr(msat0T0.pointer))),

		(C.double)(float64(muMul[X])),
		(C.double)(float64(muMul[Y])),
		(C.double)(float64(muMul[Z])),

		(C.double)(float64(KB2tempMul_mu0VgammaDtMsatMul)),
		(C.CUstream)(unsafe.Pointer(uintptr(h.Stream))),
		(C.int)(h.PartLen3D()))
}
