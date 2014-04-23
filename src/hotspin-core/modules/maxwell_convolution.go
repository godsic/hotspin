//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package modules

// 14-input, 6 output convolution for solving the general Maxwell equations.
// This implementation makes a trade-off: use less memory (good!) but more memory bandwidth (bad).
// The implementation is thus optimized for low memory usage, not for absolute top speed.
// Speed could be gained when a blocking memory recycler is in place.
//
// Author: Arne Vansteenkiste

import (
	. "hotspin-core/common"
	. "hotspin-core/engine"
	"hotspin-core/gpu"
	"hotspin-core/host"
	"math"
	"runtime"
)

// Full Maxwell Electromagnetic field solver.
// TODO: magnetic charge gives H, not B, need M
type MaxwellPlan struct {
	initialized bool             // Already initialized?
	dataSize    [3]int           // Size of the (non-zero) input data block (engine.GridSize)
	logicSize   [3]int           // Non-transformed kernel size >= dataSize (due to zeropadding)
	fftKernSize [3]int           // transformed kernel size, non-redundant parts only
	kern        *host.Array      // Real-space kernels for charge
	fftKern     [3][3]*gpu.Array // transformed kernel's non-redundant parts (only real or imag parts, or nil)
	fftMul      [3][3]complex128 // multipliers for kernel
	fftBuf      *gpu.Array       // transformed input data
	M           *gpu.Array
	fftPlan     gpu.FFTInterface // transforms input/output data
	B           *Quant
}

func (plan *MaxwellPlan) init() {
	if plan.initialized {
		return
	}
	plan.initialized = true
	e := GetEngine()
	dataSize := e.GridSize()
	logicSize := e.PaddedSize()
	Assert(len(dataSize) == 3)
	Assert(len(logicSize) == 3)

	// init size
	copy(plan.dataSize[:], dataSize)
	copy(plan.logicSize[:], logicSize)

	// init fft
	fftOutputSize := gpu.FFTOutputSize(logicSize)
	plan.fftBuf = gpu.NewArray(3, fftOutputSize)
	plan.fftPlan = gpu.NewDefaultFFT(dataSize, logicSize)

	// init M
	plan.M = gpu.NewArray(3, dataSize)

	// init fftKern
	copy(plan.fftKernSize[:], gpu.FFTOutputSize(logicSize))
	plan.fftKernSize[2] = plan.fftKernSize[2] / 2 // store only non-redundant parts
}

// Enable Demagnetizing field
func (plan *MaxwellPlan) EnableDemag(mf, msat0T0 *Quant) {
	plan.init()
	plan.loadDipoleKernel()
	runtime.GC()
}

const (
	CPUONLY = true
	GPU     = false
)

// Load dipole kernel if not yet done so.
// Required for field of electric/magnetic charge density.
func (plan *MaxwellPlan) loadDipoleKernel() {
	if plan.kern != nil {
		return
	}
	e := GetEngine()
	// DEBUG: add the kernel as orphan quant, so we can output it.
	quant := NewQuant("kern_dipole", TENS, plan.logicSize[:], FIELD, Unit(""), CPUONLY, "reduced dipole kernel")
	if DEBUG {
		e.AddQuant(quant)
	}

	kern := quant.Buffer(FIELD)
	accuracy := 6
	Kernel_Arne(plan.logicSize[:], e.CellSize(), e.Periodic(), accuracy, kern)
	plan.kern = kern
	plan.LoadKernel(kern, SYMMETRIC, PUREREAL)
}

// Calculate the magnetic field plan.B
func (plan *MaxwellPlan) UpdateB() {

	msat0T0 := GetEngine().Quant("Msat0T0")
	mf := GetEngine().Quant("mf")

	plan.M.CopyFromDevice(mf.Array())

	Debug(plan.M.Comp[X].PartLen4D(), msat0T0.Array().PartLen4D())
	if !msat0T0.Array().IsNil() {
		gpu.Mul(&plan.M.Comp[X], &plan.M.Comp[X], msat0T0.Array())
		gpu.Mul(&plan.M.Comp[Y], &plan.M.Comp[Y], msat0T0.Array())
		gpu.Mul(&plan.M.Comp[Z], &plan.M.Comp[Z], msat0T0.Array())
	}

	mMul := msat0T0.Multiplier()[0] * Mu0

	plan.ForwardFFT(plan.M)

	// Point-wise kernel multiplication
	gpu.TensSYMMVecMul(&plan.fftBuf.Comp[X], &plan.fftBuf.Comp[Y], &plan.fftBuf.Comp[Z],
		&plan.fftBuf.Comp[X], &plan.fftBuf.Comp[Y], &plan.fftBuf.Comp[Z],
		plan.fftKern[X][X], plan.fftKern[Y][Y], plan.fftKern[Z][Z],
		plan.fftKern[Y][Z], plan.fftKern[X][Z], plan.fftKern[X][Y],
		mMul,
		plan.fftKernSize[X], plan.fftKernSize[Y], plan.fftKernSize[Z],
		plan.fftBuf.Stream)

	plan.InverseFFT(plan.B.Array())

}

//// Loads a sub-kernel at position pos in the 3x3 global kernel matrix.
//// The symmetry and real/imaginary/complex properties are taken into account to reduce storage.
func (plan *MaxwellPlan) LoadKernel(kernel *host.Array, matsymm int, realness int) {

	//	for i := range kernel.Array {
	//		Debug("kernel", TensorIndexStr[i], ":", kernel.Array[i], "\n\n\n")
	//	}

	//Assert(kernel.NComp() == 9) // full tensor
	if kernel.NComp() > 3 {
		testedsymm := MatrixSymmetry(kernel)
		Debug("matsymm", testedsymm)
		// TODO: re-enable!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		//Assert(matsymm == testedsymm)
	}
	Assert(matsymm == SYMMETRIC || matsymm == ANTISYMMETRIC || matsymm == NOSYMMETRY || matsymm == DIAGONAL)

	//if FFT'd kernel is pure real or imag,
	//store only relevant part and multiply by scaling later
	scaling := [3]complex128{complex(1, 0), complex(0, 1), complex(0, 0)}[realness]
	Debug("scaling=", scaling)

	// FFT input on GPU
	logic := plan.logicSize[:]
	devIn := gpu.NewArray(1, logic)
	defer devIn.Free()

	// FFT output on GPU
	devOut := gpu.NewArray(1, gpu.FFTOutputSize(logic))
	defer devOut.Free()
	fullFFTPlan := gpu.NewDefaultFFT(logic, logic)
	defer fullFFTPlan.Free()

	// Maximum of all elements gives idea of scale.
	max := maxAbs(kernel.List)

	// FFT all components
	for k := 0; k < 9; k++ {
		i, j := IdxToIJ(k) // fills diagonal first, then upper, then lower

		// ignore off-diagonals of vector (would go out of bounds)
		if k > ZZ && matsymm == DIAGONAL {
			Debug("break", TensorIndexStr[k], "(off-diagonal)")
			break
		}

		// elements of diagonal kernel are stored in one column
		if matsymm == DIAGONAL {
			i = 0
		}

		// clear data first
		AssertMsg(plan.fftKern[i][j] == nil, "I'm afraid I can't let you overwrite that")
		AssertMsg(plan.fftMul[i][j] == 0, "Likewise")

		// ignore zeros
		if k < kernel.NComp() && IsZero(kernel.Comp[k], max) {
			Debug("kernel", TensorIndexStr[k], " == 0")
			plan.fftKern[i][j] = gpu.NilArray(1, []int{plan.fftKernSize[X], plan.fftKernSize[Y], plan.fftKernSize[Z]})
			continue
		}

		// auto-fill lower triangle if possible
		if k > XY {
			if matsymm == SYMMETRIC {
				plan.fftKern[i][j] = plan.fftKern[j][i]
				plan.fftMul[i][j] = plan.fftMul[j][i]
				continue
			}
			if matsymm == ANTISYMMETRIC {
				plan.fftKern[i][j] = plan.fftKern[j][i]
				plan.fftMul[i][j] = -plan.fftMul[j][i]
				continue
			}
		}

		// calculate FFT of kernel elementx
		Debug("use", TensorIndexStr[k])
		devIn.CopyFromHost(kernel.Component(k))
		fullFFTPlan.Forward(devIn, devOut)
		hostOut := devOut.LocalCopy()

		// extract real part of the kernel from the first quadrant (other parts are redundunt due to the symmetry properties)
		hostFFTKern := extract(hostOut)
		rescale(hostFFTKern, 1/float64(gpu.FFTNormLogic(logic)))
		plan.fftKern[i][j] = gpu.NewArray(1, hostFFTKern.Size3D)
		plan.fftKern[i][j].CopyFromHost(hostFFTKern)
		plan.fftMul[i][j] = scaling
	}

}

const zero_tolerance = 1e-12

// list is considered zero if all elements are
// at least a factorzero_tolerance smaller than max.
func IsZero(array []float64, max float64) bool {
	return (maxAbs(array) / max) < zero_tolerance
}

// maximum absolute value of all elements
func maxAbs(array []float64) (max float64) {
	for _, x := range array {
		if Abs32(x) > max {
			max = Abs32(x)
		}
	}
	return
}

// arr[i] *= scale
func rescale(arr *host.Array, scale float64) {
	list := arr.List
	for i := range list {
		list[i] = float64(float64(list[i]) * scale)
	}
}

// matrix symmetry
const (
	NOSYMMETRY    = 0  // Kij independent of Kji
	SYMMETRIC     = 1  // Kij = Kji
	DIAGONAL      = 2  // also used for vector
	ANTISYMMETRIC = -1 // Kij = -Kji
)

// Detects matrix symmetry.
// returns NOSYMMETRY, SYMMETRIC, ANTISYMMETRIC
func MatrixSymmetry(matrix *host.Array) int {
	AssertMsg(matrix.NComp() == 9, "MatrixSymmetry NComp")
	symm := true
	asymm := true
	max := 1e-100
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			scount := 0
			acount := 0
			total := 0
			idx1 := FullTensorIdx[i][j]
			idx2 := FullTensorIdx[j][i]
			comp1 := matrix.Comp[idx1]
			comp2 := matrix.Comp[idx2]
			for x := range comp1 {
				if math.Abs(float64(comp1[x])) > max {
					max = math.Abs(float64(comp1[x]))
				}
				total++
				if comp1[x] == comp2[x] {
					scount++
				}
				if comp1[x] != comp2[x] {
					//Debug(comp1[x], "!=", comp2[x])
					symm = false
					//if !asymm {
					//break
					//}
				}
				if comp1[x] == -comp2[x] {
					acount++
				}
				if comp1[x] != -comp2[x] {
					//Debug(comp1[x] ,"!= -", comp2[x])
					asymm = false
					//if !symm {
					//break
					//}
				}
			}
			Debug("max", max)
			Debug(i, j, "symm", scount, "asymm", acount, "(of", total, ")")
		}
	}
	if symm {
		return SYMMETRIC // also covers all zeros
	}
	if asymm {
		return ANTISYMMETRIC
	}
	return NOSYMMETRY
}

// data realness
const (
	PUREREAL = 0 // data is purely real
	PUREIMAG = 1 // data is purely complex
	COMPLEX  = 2 // data is full complex number
)

func (plan *MaxwellPlan) Free() {
	// TODO
}

// 	INTERNAL
// Sparse transform all 3 components.
// (FFTPlan knows about zero padding etc)
func (plan *MaxwellPlan) ForwardFFT(in *gpu.Array) {
	Assert(plan.fftBuf.NComp() == in.NComp())
	for c := range in.Comp {
		plan.fftPlan.Forward(&in.Comp[c], &plan.fftBuf.Comp[c])
	}
}

// 	INTERNAL
// Sparse backtransform
// (FFTPlan knows about zero padding etc)
func (plan *MaxwellPlan) InverseFFT(out *gpu.Array) {
	Assert(plan.fftBuf.NComp() == out.NComp())
	for c := range out.Comp {
		plan.fftPlan.Inverse(&plan.fftBuf.Comp[c], &out.Comp[c])
	}
}

// Extract real or imaginary parts, copy them from src to dst.
// In the meanwhile, check if the other parts are nearly zero
// and scale the kernel to compensate for unnormalized FFTs.
// real_imag = 0: real parts
// real_imag = 1: imag parts
func extract(src *host.Array) *host.Array {

	sx := src.Size3D[X]/2 + 1 // antisymmetric
	sy := src.Size3D[Y]/2 + 1 // antisymmetric
	sz := src.Size3D[Z] / 2   // only real parts should be stored, the value of the imaginary part should stay below the zero threshould
	dst := host.NewArray(src.NComp(), []int{sx, sy, sz})

	dstArray := dst.Array
	srcArray := src.Array

	// Normally, the FFT'ed kernel is purely real because of symmetry,
	// so we only store the real parts...
	maxImg := float64(0.)
	maxReal := float64(0.)
	for c := range dstArray {
		for k := range dstArray[c] {
			for j := range dstArray[c][k] {
				for i := range dstArray[c][k][j] {
					dstArray[c][k][j][i] = srcArray[c][k][j][2*i]
					if Abs32(srcArray[c][k][j][2*i+1]) > maxImg {
						maxImg = Abs32(srcArray[c][k][j][2*i+1])
					}
					if Abs32(srcArray[c][k][j][2*i+0]) > maxReal {
						maxReal = Abs32(srcArray[c][k][j][2*i+0])
					}
				}
			}
		}
	}
	// ...however, we check that the imaginary parts are nearly zero,
	// just to be sure we did not make a mistake during kernel creation.
	Debug("FFT Kernel max real part", 0, ":", maxReal)
	Debug("FFT Kernel max imag part", 1, ":", maxImg)
	Debug("FFT Kernel max imag/real part=", maxImg/maxReal)
	if maxImg/maxReal > 1e-12 { // TODO: is this reasonable?
		panic(BugF("FFT Kernel max bad/good part=", maxImg/maxReal))
	}
	return dst
}
