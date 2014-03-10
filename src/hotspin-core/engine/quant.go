//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// This file implements physical quantities represented by
// either scalar, vector or tensor fields in time and space.
// Author: Arne Vansteenkiste.

import (
	"fmt"
	. "hotspin-core/common"
	"hotspin-core/gpu"
	"hotspin-core/host"
	"math"
	"sync"
)

// A quantity represents a scalar/vector/tensor field,value or mask.
//
// * By "value" we mean a single, space-independent scalar,vector or tensor.
// * By "field" we mean a space-dependent field of scalars, vectors or tensors.
// * By "mask" we mean the point-wise multiplication of a field by a value.
//
// Typically a mask represents A(r) * f(t), a pointwise multiplication
// of an N-vector function of space A(r) by an N-vector function of time f(t).
// A(r) is an array, f(t) is the multiplier which will be updated every time step.
// When a mask's array contains NULL pointers for each gpu, the mask is independent of space.
// The array is then interpreted as 1(r), the unit field. In this way, masks that happen to be
// constant values in space (homogeneous) can be efficiently represented.
// TODO: deduplicate identical mask arrays by setting identical pointers?
//
// Quantities are the nodes of an acyclic graph representing the differential
// equation to be solved.
type Quant struct {
	name       string    // Unique identifier
	array      gpu.Array // Underlying array of dimensionless values typically of order 1. Holds nil pointers for space-independent quantities.
	multiplier []float64 // Point-wise multiplication coefficients for array, dimensionfull.
	nComp      int       // Number of components. Defines whether it is a SCALAR, VECTOR, TENSOR,...
	upToDate   bool      // Flags if this quantity needs to be updated
	updater    Updater   // Called to update this quantity
	//invalidator Invalidator       // Called before each Invalidate()
	verifier    func(q *Quant)    // Called to verify user input
	children    map[string]*Quant // Quantities this one depends on, indexed by name
	parents     map[string]*Quant // Quantities that depend on this one, indexed by name
	desc        string            // Human-readable description
	unit        Unit              // Unit of the multiplier value, e.g. A/m.
	kind        QuantKind         // VALUE, FIELD or MASK
	updates     int               // Number of times the quantity has been updated (for debuggin)
	invalidates int               // Number of times the quantity has been invalidated (for debuggin)
	cpuOnly     bool              // true if quantity exists only in CPU RAM, not on GPU
	buffer      *host.Array       // Host buffer for copying from/to the GPU array
	bufUpToDate bool              // Flags if the buffer (in RAM) needs to be updated
	bufXfers    int               // Number of times it has been copied from GPU
	bufKind     QuantKind         // The format of the buffer
	timer       Timer             // Debug/benchmarking
	bufMutex    sync.RWMutex
}

//____________________________________________________________________ init

// Returns a new quantity. See Quant.init(). TODO: make desc obligatory
func NewQuant(name string, nComp int, size3D []int, kind QuantKind, unit Unit, cpuOnly bool, desc ...string) *Quant {
	q := new(Quant)
	q.init(name, nComp, size3D, kind, unit, cpuOnly, desc...)
	return q
}

// Number of components.
const (
	SCALAR   = 1 // Number
	VECTOR   = 3 // Vector
	SYMMTENS = 6 // Symmetric tensor
	TENS     = 9 // General tensor
)

// Initiates a field with nComp components and array size size3D.
// When size3D == nil, the field is space-independent (homogeneous) and the array will
// hold NULL pointers for each of the GPU parts.
// indicating this quantity should not be post-multiplied.
// multiply = true
func (q *Quant) init(name string, nComp int, size3D []int, kind QuantKind, unit Unit, cpuOnly bool, desc ...string) {
	Assert(nComp > 0)
	Assert(size3D == nil || len(size3D) == 3)
	if cpuOnly { // no mask games on CPU.
		Assert(kind == FIELD)
	}

	q.name = name
	q.nComp = nComp
	q.kind = kind
	q.unit = unit
	q.cpuOnly = cpuOnly
	q.initChildrenParents()

	switch kind {
	// A FIELD is calculated by mumax itself, not settable by the user.
	// So it should not have a multiplier, but always have allocated storage.
	case FIELD:
		alloc := !cpuOnly
		q.array.Init(nComp, size3D, alloc)
		q.multiplier = ones(nComp)
	// A MASK should always have a value (stored in the multiplier).
	// We initialize it to zero. The space-dependent mask is optinal
	// and not yet allocated.
	case MASK:
		q.array.Init(nComp, size3D, false)
		q.multiplier = zeros(nComp)
	// A VALUE is space-independent and thus does not have allocated storage.
	// The value is stored in the multiplier and initialized to zero.
	case VALUE:
		//q.array = nil
		q.multiplier = zeros(nComp)
	default:
		panic(Bug("Quant.init kind"))
	}

	if cpuOnly {
		q.allocBuffer() // Allocate CPU storage
	}

	// concatenate desc strings
	buf := ""
	for i, str := range desc {
		if i > 0 {
			str += " "
		}
		buf += str
	}
	q.desc = buf
}

// Safely sets the updater.
func (q *Quant) SetUpdater(u Updater) {
	if q.updater != nil {
		panic(Bug(fmt.Sprint("Quant.SetUpdater:", q.Name(), "updater already set.")))
	}
	q.updater = u
}

// Safely sets the invalidator.
//func (q *Quant) SetInvalidator(v Invalidator) {
//	if q.invalidator != nil {
//		panic(Bug(fmt.Sprint("Quant.SetInvalidator:", q.Name(), "invalidator already set.")))
//	}
//	q.invalidator = v
//}

// Gets the updater
func (q *Quant) Updater() Updater {
	return q.updater
}

func (q *Quant) initChildrenParents() {
	q.children = make(map[string]*Quant)
	q.parents = make(map[string]*Quant)
}

// Quantity representing a single component of the original,
// with shared underlying storage.
// The returned Quant's name and dependencies still have to be set.
func (orig *Quant) Component(comp int) *Quant {
	q := new(Quant)

	q.nComp = 1
	q.kind = orig.kind

	q.array.Assign(&(orig.array.Comp[comp]))      // share storage with parent
	q.multiplier = orig.multiplier[comp : comp+1] // share storage with parent
	if orig.cpuOnly {
		q.cpuOnly = true
		q.buffer = orig.buffer.Component(comp) // share storage with parent
	}

	q.initChildrenParents()

	q.unit = orig.unit
	return q
}

// array with n 1's.
func ones(n int) []float64 {
	ones := make([]float64, n)
	for i := range ones {
		ones[i] = 1
	}
	return ones
}

// array with n 0's.
func zeros(n int) []float64 {
	zeros := make([]float64, n)
	for i := range zeros {
		zeros[i] = 0
	}
	return zeros
}

//____________________________________________________________________ set

// Set the multiplier of a MASK or the value of a VALUE
func (q *Quant) SetValue(val []float64) {
	//Debug("SetValue", q.name, val)
	//~ checkKinds(q, MASK, VALUE)
	checkComp(q, len(val))
	if q.kind == MASK || q.kind == VALUE {
		for i, v := range val {
			q.multiplier[i] = v
		}
	} else if q.kind == FIELD {
		q.multiplier = ones(q.nComp)
		// use q.Buffer instead?
		tempField := host.NewArray(q.nComp, q.array.Size3D())
		for c := 0; c < q.nComp; c++ {
			for i := 0; i < q.array.Size3D()[X]; i++ {
				for j := 0; j < q.array.Size3D()[Y]; j++ {
					for k := 0; k < q.array.Size3D()[Z]; k++ {
						tempField.Array[c][i][j][k] = float64(val[c])
					}
				}
			}
		}
		q.SetField(tempField)
		// not sure whenever tempBuffer will be destroyed by the GC?
	} else {
		panic(InputErr(q.name + " is not " + MASK.String() + " or " + FIELD.String() + " or " + VALUE.String() + " but " + q.kind.String()))
	}

	q.Verify()
	q.Invalidate() //!
}

// Convenience method for SetValue([]float64{val})
func (q *Quant) SetScalar(val float64) {
	checkKind(q, VALUE)
	q.multiplier[0] = val
	q.Invalidate() //!
}

// Sets one component of a VALUE quantity.
func (q *Quant) SetComponent(comp int, val float64) {
	checkKind(q, VALUE)
	q.multiplier[comp] = val
	q.Invalidate() //!
}

// Sets a space-dependent field.
func (q *Quant) SetField(field *host.Array) {
	checkKind(q, FIELD)
	q.Array().CopyFromHost(field)
	q.Invalidate() //!
}

// Sets the space-dependent mask array.
// Allocates GPU storage when needed.
func (q *Quant) SetMask(field *host.Array) {
	checkKind(q, MASK)
	q.assureAlloc()
	//Debug(q.Name(), q.Array())
	q.Array().CopyFromHost(field)
	q.Invalidate() //!
}

// 	sum += parent
func (sum *Quant) Add(parent *Quant) {
	invalidated := false
	for c := 0; c < sum.NComp(); c++ {
		parComp := parent.array.Component(c)
		parMul := parent.multiplier[c]
		if parMul == 0 {
			continue
		}
		sumMul := sum.multiplier[c]
		sumComp := sum.array.Component(c)                  // does not alloc
		gpu.Madd(sumComp, sumComp, parComp, parMul/sumMul) // divide by sum's multiplier!
		invalidated = true
	}
	if invalidated {
		sum.Invalidate()
	}
}

//____________________________________________________________________ get

// Assuming the quantity represent a scalar value, return it as a number.
func (q *Quant) Scalar() float64 {
	q.Update()
	if q.IsSpaceDependent() {
		panic(InputErr(q.Name() + " is space-dependent, can not return it as a scalar value"))
	}
	return q.multiplier[0]
}

// Gets the name
func (q *Quant) Name() string {
	return q.name
}

// Gets the name + [unit]
func (q *Quant) FullName() string {
	unit := string(q.unit)
	if unit == "" {
		return q.name
	}
	return q.name + " [" + unit + "]"
}

// Gets the number of components
func (q *Quant) NComp() int {
	return q.nComp
}

// Grid size of the quantity,
// not necessarily the engine's grid size
func (q *Quant) Size3D() []int {
	return q.array.Size3D()
}

// The quantities unit.
func (q *Quant) Unit() Unit {
	return q.unit
}

// Gets the GPU array.
func (q *Quant) Array() *gpu.Array {
	return &(q.array)
}

// True if the quantity varies in space.
func (q *Quant) IsSpaceDependent() bool {
	return q.kind == FIELD || q.kind == MASK && !q.array.IsNil()
}

//
func (q *Quant) Multiplier() []float64 {
	return q.multiplier
}

func (q *Quant) Kind() QuantKind {
	return q.kind
}

// Transfers the quantity from GPU to host. The quantities host buffer
// is allocated when needed. The transfer is only done when needed, i.e.,
// when bufferUpToDate == false. Multiplies by the multiplier and handles masks correctly.
// Does not Update().
func (q *Quant) Buffer(kind QuantKind) *host.Array {
	if q.cpuOnly || (q.bufUpToDate && kind == q.bufKind) {
		//Debug("buffer of", q.Name(), q.buffer.Array)
		return q.buffer
	}
	//Debug("XFer", q.Name())

	q.bufMutex.Lock()

	// allocate if needed
	array := q.Array()
	if q.buffer == nil {
		q.allocBuffer()
	}
	// copy
	buffer := q.buffer
	buffer.Pin()

	if array.IsNil() {
		for c := range buffer.Comp {
			comp := buffer.Comp[c]
			for i := range comp {
				switch kind {
				case FIELD:
					comp[i] = float64(q.multiplier[c])
				case MASK:
					comp[i] = float64(1.0)
				}
			}
		}
	} else {
		q.array.CopyToHost(q.buffer)
		q.bufXfers++
		switch kind {
		case FIELD:
			for c := range buffer.Comp {
				if q.multiplier[c] != 1 {
					comp := buffer.Comp[c]
					for i := range comp {
						comp[i] *= float64(q.multiplier[c]) // multiply by multiplier if not 1
					}
				}
			}
		case MASK:
		}
	}
	q.bufUpToDate = true
	q.bufKind = kind
	q.bufMutex.Unlock()
	return q.buffer
}

func (q *Quant) allocBuffer() {
	if q.buffer != nil {
		panic(Bug("Buffer already allocated"))
	}
	q.buffer = host.NewArrayPinned(q.NComp(), q.Array().Size3D())
}

//____________________________________________________________________ tree walk

// If q.upToDate is false, update this node recursively.
// First Update all parents (on which this node depends),
// and then call Quant.updateSelf.Update().
// upToDate is set true.
// See: Invalidate()
func (q *Quant) Update() {
	// update parents first
	for _, p := range q.parents {
		p.Update()
	}

	// now update self
	//Log("actually update " + q.Name())
	if !q.upToDate {
		q.timer.Start()
		if q.updater != nil {
			q.updater.Update()
		}
		q.timer.Stop()
		q.updates++
	}

	// verify if new value is OK
	if q.verifier != nil {
		q.Verify()
	}
	for _, m := range q.multiplier {
		if math.IsNaN(m) {
			panic("NaN")
			//panic(BugF(q.Name(), " is NaN.", "timestep", engine.step.multiplier[0])) // crashes...
		}
	}
	q.upToDate = true
}

// Opposite of Update. Sets upToDate flag of this node and
// all its children (which depend on this node) to false.
func (q *Quant) Invalidate() {
	// invalidator is called before actual invalidate!
	//	if q.invalidator != nil {
	//		q.invalidator.Invalidate()
	//	}

	if q.upToDate {
		q.invalidates++
	}
	q.upToDate = false
	q.bufUpToDate = false
	q.bufKind = FIELD
	for _, c := range q.children {
		c.Invalidate()
	}
}

// Verifies if the quantity's value makes sense.
// Called, e.g., after the user set a value.
// TODO: not used consistently
func (q *Quant) Verify() {
	if q.verifier != nil {
		q.verifier(q)
	}
}

// Sets a verifier func, called each time the
// user attempts to set the quantity.
func (q *Quant) SetVerifier(f func(*Quant)) {
	if q.verifier != nil {
		panic(Bug(q.Name() + " verifier already set"))
	}
	q.verifier = f
}

// Gets the updater
func (q *Quant) GetUpdater() Updater {
	return q.updater
}

//___________________________________________________________

// INTERNAL: in case of a MASK, make sure the underlying array is allocted.
// Used, e.g., when a space-independent mask gets replaced by a space-dependent one.
func (q *Quant) assureAlloc() {
	pointer := q.Array().DevicePtr()
	if pointer == 0 {
		Debug("assureAlloc: " + q.Name())
		q.Array().Alloc()
		//Debug(q.Name(), q.Array())
	}
}

// Checks if the quantity has the specified kind
// Panics if check fails.
func checkKind(q *Quant, kind QuantKind) {
	if q.kind != kind {
		panic(InputErr(q.name + " is not " + kind.String() + " but " + q.kind.String()))
	}
}

// Checks if the quantity has one of the specified kinds.
// Panics if check fails.
func checkKinds(q *Quant, kind1, kind2 QuantKind) {
	if q.kind != kind1 && q.kind != kind2 {
		panic(InputErr(q.name + " is not " + kind1.String() + " or " + kind2.String() + " but " + q.kind.String()))
	}
}

// Checks if the quantity has ncomp components.
// Panics if check fails.
func checkComp(q *Quant, ncomp int) {
	if ncomp != q.nComp {
		panic(InputErr(fmt.Sprint(q.Name(), " has ", q.nComp, " components, but ", ncomp, " are provided.")))
	}
}

func (q *Quant) String() string {
	return fmt.Sprint(q.Name(), q.Buffer(FIELD).Array)
}

func (q *Quant) SetUpToDate(status bool) {
	q.upToDate = status
}

func (q *Quant) CopyFromQuant(qin *Quant) {
	checkKind(q, qin.Kind())
	checkComp(q, qin.NComp())
	if q.unit != qin.unit {
		panic(InputErrF("Units mismatch:", q.FullName(), "!=", qin.FullName()))
	}

	for i := 0; i < q.nComp; i++ {
		q.multiplier[i] = qin.multiplier[i]
	}

	q.array.CopyFromDevice(&(qin.array))
	q.Invalidate()
}

//// If the quantity represents a space-dependent field, return a host copy of its value.
//// Call FreeBuffer() to recycle it.
//func (q *Quant) FieldValue() *host.Array {
//	a := q.array
//	buffer := NewBuffer(a.NComp(), a.Size3D())
//	q.array.CopyToHost(buffer)
//	return buffer
//}

//
//func (f *Field) Free() {
//	f.array.Free()
//	f.multiplier = nil
//	f.name = ""
//}
//
//func (f *Field) Name() string {
//	return f.name
//}
//
//// Number of components of the field values.
//// 1 = scalar, 3 = vector, etc.
//func (f *Field) NComp() int {
//	return len(f.multiplier)
//}
//
//func (f *Field) IsSpaceDependent() bool {
//	return !f.array.IsNil()
//}
//
//
//func (f *Field) String() string {
//	str := f.Name() + "(" + fmt.Sprint(f.NComp()) + "-vector "
//	if f.IsSpaceDependent() {
//		str += "field"
//	} else {
//		str += "value"
//	}
//	str += ")"
//	return str
//}
//
