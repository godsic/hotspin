//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

import (
	//~ . "hotspin-core/common"
	"math"
	"hotspin-core/gpu"
	"hotspin-core/host"
)

// Update FFT of quantity
type FFTUpdater struct {
	in   *Quant
	out  *Quant
	q    *gpu.Array
	win  *gpu.Array
	buff *gpu.Array
	plan gpu.FFTInterface
	norm float64
}

func NewFFTUpdater(qin, qout *Quant) *FFTUpdater {

	u := new(FFTUpdater)
	u.in = qin
	u.out = qout

	meshSize := engine.GridSize()

	u.win = gpu.NewArray(1, meshSize)
	u.win.CopyFromHost(genWindow(meshSize))

	u.q = gpu.NewArray(qin.NComp(), meshSize)

	u.norm = 1.0 / float64(gpu.FFTNormLogic(meshSize))

	u.plan = gpu.NewDefaultFFT(meshSize, meshSize)

	engine.Depends(qout.Name(), qin.Name())

	return u
}

func (u *FFTUpdater) Update() {

	window := u.win
	qin := u.in.Array()
	qout := u.out.Array()
	q := u.q

	COMP := u.in.NComp()

	//~ apply windowing
	for ii := 0; ii < COMP; ii++ {
		gpu.Mul(q.Component(ii), qin.Component(ii), window)
	}
    //~ dot fft
	for ii := 0; ii < COMP; ii++ {
        u.out.Multiplier()[ii] *= u.norm
		u.plan.Forward(q.Component(ii), qout.Component(ii))
	}
}

func gauss(arg, w float64) float64 {
	iw := 1.0 / w
	return math.Exp(-0.5 * iw * iw * (2.0*arg - 1.0) * (2.0*arg - 1.0))
}

func blackmanNuttall(arg float64) float64 {
	return 0.3635819 - 0.4891775*math.Cos(2.0*math.Pi*arg) + 0.1365995*math.Cos(4.0*math.Pi*arg) - 0.0106411*math.Cos(6.0*math.Pi*arg)
}

func hamming(arg float64) float64 {
	return 0.53836 - 0.46164*math.Cos(2.0*math.Pi*arg)
}

func genWindow(size []int) *host.Array {
	window := host.NewArray(1, size)
	for i := 0; i < size[0]; i++ {
		val0 := 1.0
		if size[0] > 16 {
			arg0 := float64(i) / float64(size[0]-1)
			val0 = gauss(arg0, 0.4)
		}
		for j := 0; j < size[1]; j++ {
			val1 := 1.0
			if size[1] > 16 {
				arg1 := float64(j) / float64(size[1]-1)
				val1 = gauss(arg1, 0.4)
			}
			for k := 0; k < size[2]; k++ {
				val2 := 1.0
				if size[2] > 16 {
					arg2 := float64(k) / float64(size[2]-1)
					val2 = gauss(arg2, 0.4)
				}
				window.Array[0][i][j][k] = float64(val0 * val1 * val2)
			}
		}

	}
	return window
}
