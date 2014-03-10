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
	"fmt"
	"math"
	. "hotspin-core/common"
	. "hotspin-core/engine"
	"hotspin-core/gpu"
	"hotspin-core/host"
	"runtime"
	"unsafe"
)

// Full Maxwell Electromagnetic field solver.
// TODO: magnetic charge gives H, not B, need M
type MaxwellPlan struct {
	initialized bool             // Already initialized?
	dataSize    [3]int           // Size of the (non-zero) input data block (engine.GridSize)
	logicSize   [3]int           // Non-transformed kernel size >= dataSize (due to zeropadding)
	fftKernSize [3]int           // transformed kernel size, non-redundant parts only
	kern        [3]*host.Array   // Real-space kernels for charge, dipole, rotor
	fftKern     [7][3]*gpu.Array // transformed kernel's non-redundant parts (only real or imag parts, or nil)
	fftMul      [7][3]complex128 // multipliers for kernel
	fftBuffer   gpu.Array        // transformed input data
	fftOut      gpu.Array        // transformed output data (3-comp)
	fftPlan     gpu.FFTInterface // transforms input/output data
	EInput      [7]*gpu.Array    // input quantities for electric field (rho, Px, Py, Pz, 0, 0, 0)
	BInput      [7]*gpu.Array    // input quantities for magnetic field (rhoB, mx, my, mz, jx, jy, jz)
	EInMul      [7]float64       // E input multipliers (epsillon0 etc)
	BInMul      [7]float64       // B input multipliers (mu0 etc)
	//BExt, EExt  *Quant           // external B/E field
	dBdt, dEdt *Quant // time derivative of field
	E, B       *Quant // E/B fields
	// TODO: time derivatives could be taken in FFT space, but this complicates external fields
	//fftE1, fftE2 *gpu.Array       // previous FFT E fields for time derivative
	//fftB1, fftB2 *gpu.Array       // previous FFT B fields for time derivative
	//time1, time2 float64          // time of previous fields for derivative
}

// Index for kern
const (
	CHARGE = 0
	DIPOLE = 1
	ROTOR  = 2
)

// Index for EInput, BInput, EInMul, BInMul
const (
	RHO = 0
	PX  = 1
	MX  = 1
	PY  = 2
	MY  = 2
	PZ  = 3
	MZ  = 3
	JX  = 4
	JY  = 5
	JZ  = 6
)

//
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
	plan.fftBuffer.Init(1, fftOutputSize, gpu.DO_ALLOC) // TODO: recycle
	plan.fftOut.Init(3, fftOutputSize, gpu.DO_ALLOC)    // TODO: recycle
	plan.fftPlan = gpu.NewDefaultFFT(dataSize, logicSize)

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
	if plan.kern[DIPOLE] != nil {
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
	plan.kern[DIPOLE] = kern
	plan.LoadKernel(kern, 1, SYMMETRIC, PUREREAL)
}

// Calculate the electric field plan.E
func (plan *MaxwellPlan) UpdateE() {
	plan.update(&plan.EInput, &plan.EInMul, plan.E.Array(), nil) //plan.EExt)
}

// Calculate the magnetic field plan.B
func (plan *MaxwellPlan) UpdateB() {
	var normM *gpu.Array
	hasMsat := 0

	msat0T0 := GetEngine().Quant("Msat0T0")
	mf := GetEngine().Quant("mf")

	if !msat0T0.Array().IsNil() {
		hasMsat = 1

		sx := mf.Array().Size3D()[X]
		sy := mf.Array().Size3D()[Y]
		sz := mf.Array().Size3D()[Z]

		normM = gpu.NewArray(3, []int{sx, sy, sz})

		gpu.Mul(normM.Component(X), mf.Array().Component(X), msat0T0.Array())
		gpu.Mul(normM.Component(Y), mf.Array().Component(Y), msat0T0.Array())
		gpu.Mul(normM.Component(Z), mf.Array().Component(Z), msat0T0.Array())

		plan.BInput[MX] = normM.Component(X)
		plan.BInput[MY] = normM.Component(Y)
		plan.BInput[MZ] = normM.Component(Z)
	} else {
		plan.BInput[MX] = mf.Array().Component(X)
		plan.BInput[MY] = mf.Array().Component(Y)
		plan.BInput[MZ] = mf.Array().Component(Z)
	}

	plan.BInMul[MX] = msat0T0.Multiplier()[0] * Mu0
	plan.BInMul[MY] = msat0T0.Multiplier()[0] * Mu0
	plan.BInMul[MZ] = msat0T0.Multiplier()[0] * Mu0

	plan.update(&plan.BInput, &plan.BInMul, plan.B.Array(), nil)
	if hasMsat == 1 {
		normM.Free()
	}
}

// calculate E or B
func (plan *MaxwellPlan) update(in *[7]*gpu.Array, inMul *[7]float64, out *gpu.Array, ext *Quant) {
	fftBuf := &plan.fftBuffer
	fftOut := &plan.fftOut
	fftOut.Zero() // TODO: smarter kernel mul that does not read/add/read/add/...
	needInverseFFT := false
	for i := 0; i < 7; i++ {
		if in[i] == nil {
			continue
		}
		//fmt.Println("in", i, in[i].LocalCopy().Array)
		plan.ForwardFFT(in[i])
		for j := 0; j < 3; j++ {
			if plan.fftKern[i][j] == nil {
				continue
			}
			// Point-wise kernel multiplication
			mul := complex128(complex(inMul[i], 0) * plan.fftMul[i][j])
			//Debug(i, j, "mul", mul)
			gpu.CMaddAsync(&fftOut.Comp[j], mul, plan.fftKern[i][j], fftBuf, fftOut.Stream)
			needInverseFFT = true
			fftOut.Stream.Sync()
		}
	}
	if needInverseFFT {
		plan.InverseFFT(out)
	} else {
		out.Zero()
	}
	// add external field
	if ext != nil {
		for c := 0; c < 3; c++ {
			mul := ext.Multiplier()[c]
			if mul != 0 {
				gpu.Madd(out.Component(c), out.Component(c), ext.Array().Component(c), mul)
			}
		}
	}
}

//// Loads a sub-kernel at position pos in the 3x7 global kernel matrix.
//// The symmetry and real/imaginary/complex properties are taken into account to reduce storage.
func (plan *MaxwellPlan) LoadKernel(kernel *host.Array, pos int, matsymm int, realness int) {

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
		AssertMsg(plan.fftKern[i+pos][j] == nil, "I'm afraid I can't let you overwrite that")
		AssertMsg(plan.fftMul[i+pos][j] == 0, "Likewise")

		// ignore zeros
		if k < kernel.NComp() && IsZero(kernel.Comp[k], max) {
			Debug("kernel", TensorIndexStr[k], " == 0")
			continue
		}

		// auto-fill lower triangle if possible
		if k > XY {
			if matsymm == SYMMETRIC {
				plan.fftKern[i+pos][j] = plan.fftKern[j+pos][i]
				plan.fftMul[i+pos][j] = plan.fftMul[j+pos][i]
				continue
			}
			if matsymm == ANTISYMMETRIC {
				plan.fftKern[i+pos][j] = plan.fftKern[j+pos][i]
				plan.fftMul[i+pos][j] = -plan.fftMul[j+pos][i]
				continue
			}
		}

		// calculate FFT of kernel element
		Debug("use", TensorIndexStr[k])
		devIn.CopyFromHost(kernel.Component(k))
		fullFFTPlan.Forward(devIn, devOut)
		hostOut := devOut.LocalCopy()

		// extract real or imag parts
		hostFFTKern := extract(hostOut, realness)
		rescale(hostFFTKern, 1/float64(gpu.FFTNormLogic(logic)))
		plan.fftKern[i+pos][j] = gpu.NewArray(1, hostFFTKern.Size3D)
		plan.fftKern[i+pos][j].CopyFromHost(hostFFTKern)
		plan.fftMul[i+pos][j] = scaling
	}

	// debug
	if DEBUG {
		var dbg [7][3]string
		for i := range dbg {
			for j := range dbg[i] {
				dbg[i][j] = fmt.Sprint(unsafe.Pointer(plan.fftKern[i][j]), "*", plan.fftMul[i][j])
			}
		}
		Debug("maxwell convplan kernel:", dbg)
	}
}

const zero_tolerance = 1e-5

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
	Assert(plan.fftBuffer.NComp() == in.NComp())
	Assert(plan.fftBuffer.NComp() == 1)
	//for c := range in.Comp {
	plan.fftPlan.Forward(in, &plan.fftBuffer)
	//}
}

// 	INTERNAL
// Sparse backtransform
// (FFTPlan knows about zero padding etc)
func (plan *MaxwellPlan) InverseFFT(out *gpu.Array) {
	Assert(plan.fftOut.NComp() == out.NComp())
	for c := range out.Comp {
		plan.fftPlan.Inverse(&plan.fftOut.Comp[c], &out.Comp[c])
	}
}

//func (conv *Conv73Plan) SelfTest() {
//	Debug("FFT self-test")
//	rng := rand.New(rand.NewSource(0))
//	size := conv.dataSize[:]
//
//	in := NewArray(1, size)
//	defer in.Free()
//	arr := in.LocalCopy()
//	a := arr.List
//	for i := range a {
//		a[i] = 2*rng.Float32() - 1
//		if a[i] == 0 {
//			a[i] = 1
//		}
//	}
//	in.CopyFromHost(arr)
//
//	out := NewArray(1, size)
//	defer out.Free()
//
//	conv.ForwardFFT(in)
//	conv.InverseFFT(out)
//
//	b := out.LocalCopy().List
//	norm := float64(1 / float64(FFTNormLogic(conv.logicSize[:])))
//	var maxerr float64
//	for i := range a {
//		if Abs32(a[i]-b[i]*norm) > maxerr {
//			maxerr = Abs32(a[i] - b[i]*norm)
//		}
//	}
//	Debug("FFT max error:", maxerr)
//	if maxerr > 1e-3 {
//		panic(BugF("FFT self-test failed, max error:", maxerr, "\nPlease use a different grid size of FFT type."))
//	}
//	runtime.GC()
//}

// Extract real or imaginary parts, copy them from src to dst.
// In the meanwhile, check if the other parts are nearly zero
// and scale the kernel to compensate for unnormalized FFTs.
// real_imag = 0: real parts
// real_imag = 1: imag parts
func extract(src *host.Array, realness int) *host.Array {
	if realness == COMPLEX {
		return src
	}

	sx := src.Size3D[X]
	sy := src.Size3D[Y]
	sz := src.Size3D[Z] / 2 // only real/imag parts
	dst := host.NewArray(src.NComp(), []int{sx, sy, sz})

	dstList := dst.List
	srcList := src.List

	// Normally, the FFT'ed kernel is purely real because of symmetry,
	// so we only store the real parts...
	maxbad := float64(0.)
	maxgood := float64(0.)
	other := 1 - realness
	for i := range dstList {
		dstList[i] = srcList[2*i+realness]
		if Abs32(srcList[2*i+other]) > maxbad {
			maxbad = Abs32(srcList[2*i+other])
		}
		if Abs32(srcList[2*i+realness]) > maxgood {
			maxgood = Abs32(srcList[2*i+realness])
		}
	}
	// ...however, we check that the imaginary parts are nearly zero,
	// just to be sure we did not make a mistake during kernel creation.
	Debug("FFT Kernel max part", realness, ":", maxgood)
	Debug("FFT Kernel max part", other, ":", maxbad)
	Debug("FFT Kernel max bad/good part=", maxbad/maxgood)
	if maxbad/maxgood > 1e-5 { // TODO: is this reasonable?
		panic(BugF("FFT Kernel max bad/good part=", maxbad/maxgood))
	}
	return dst
}
