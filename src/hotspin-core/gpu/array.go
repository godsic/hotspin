//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// This file implements 3-dimensional arrays of N-vectors distributed over multiple GPUs.
// Author: Arne Vansteenkiste

import (
	cu "cuda/driver"
	"fmt"
	. "hotspin-core/common"
	"hotspin-core/host"
	"math"
	"unsafe"
)

// A MuMax Array represents a 3-dimensional array of N-vectors.
//
// Layout example for a (3,4) vsplice on 2 GPUs:
// 	GPU0: X0 X1  Y0 Y1 Z0 Z1
// 	GPU1: X2 X3  Y2 Y3 Z2 Z3
type Array struct {
	pointer   cu.DevicePtr // Pointers to array parts on each GPU.
	_size     [4]int       // INTERNAL {components, size0, size1, size2}
	size4D    []int        // {components, size0, size1, size2}
	size3D    []int        // {size0, size1, size2}
	_partSize [3]int       // INTERNAL
	partSize  []int        // size of the parts of the array on each gpu.
	partLen4D int          // total number of floats per GPU
	partLen3D int          // total number of floats per GPU for one component
	Stream                 // GPU stream for general use with this array
	Comp      []Array      // X,Y,Z components as arrays
}

// Initializes the array to hold a field with the number of components and given size.
// 	Init(3, 1000) // gives an array of 1000 3-vectors
// 	Init(1, 1000) // gives an array of 1000 scalars
// 	Init(6, 1000) // gives an array of 1000 6-vectors or symmetric tensors
// Storage is allocated only if alloc == true.
func (a *Array) Init(components int, size3D []int, alloc bool) {
	a.initSize(components, size3D)

	a.Stream = NewStream()
	if alloc {
		a.Alloc()
	}

	// initialize component arrays
	a.initComp()

}

// initialize component arrays
func (a *Array) initComp() {
	a.Comp = make([]Array, a.NComp())
	for c := range a.Comp {
		a.Comp[c].initSize(1, a.Size3D())
		a.Comp[c].Stream = NewStream()
		a.Comp[c].Comp = nil

	}
	a.initCompPtrs()
}

// a = other
// (accessible from packages where Array is not assignable)
func (a *Array) Assign(other *Array) {
	a.pointer = other.pointer
	a._size = other._size
	a.size4D = other.size4D
	a.size3D = other.size3D
	a._partSize = other._partSize
	a.partSize = other.partSize
	a.partLen4D = other.partLen4D
	a.partLen3D = other.partLen3D
	a.Stream = other.Stream
	a.Comp = other.Comp
}

// Lets the pointers of an already initialized, but not allocated array (shared)
// point to an allocated array (original) possibly with an offset.
func (shared *Array) PointTo(original *Array, offset int) {
	Assert(shared.Len()+offset <= original.Len())
	shared.pointer = cu.DevicePtr(ArrayOffset(uintptr(original.pointer), offset))
}

//// Returns a new array that shares storage with the original array.
//// The new array's total number of elements should fit in the original,
//// but all other sizes may be arbitrary.
//// Possibly dangerous to use. Typically used to save memory.
//func (original *Array) SharedArray(nComp int, size []int) *Array {
//	shared := new(Array)
//	shared.pointer = original.pointer
//	shared.initSize(nComp, size)
//	shared.initComp()
//	return shared
//}

// Parameters for Array.Init()
const (
	DO_ALLOC   = true
	DONT_ALLOC = false
)

// INTERNAL
// initialize pointers to the component arrays.
// called after the GPU storage has been changed.
func (a *Array) initCompPtrs() {
	for c := range a.Comp {
		start := c * a.partLen3D
		a.Comp[c].pointer = cu.DevicePtr(offset(uintptr(a.pointer), start*SIZEOF_FLOAT))
	}
}

// INTERNAL
// initialize the sizes
func (a *Array) initSize(components int, size3D []int) {
	Assert(components > 0)
	Assert(len(size3D) == 3)
	length3D := Prod(size3D)
	Assert(length3D > 0)
	a.partLen4D = components * length3D
	a.partLen3D = length3D

	a._size[0] = components
	for i := range size3D {
		a._size[i+1] = size3D[i]
	}
	a.size4D = a._size[:]
	a.size3D = a._size[1:]
	// Slice along the J-direction
	a._partSize[X] = a.size3D[X]
	a._partSize[Y] = a.size3D[Y]
	a._partSize[Z] = a.size3D[Z]
	a.partSize = a._partSize[:]

}

// Returns an array which holds a field with the number of components and given size.
func NewArray(components int, size3D []int) *Array {
	t := new(Array)
	t.Init(components, size3D, DO_ALLOC)
	return t
}

// Returns an array without underlying storage.
// This is used for space-independent quantities. These pass
// a multiplier value and a null pointer for each GPU.
// A NilArray already has null pointers for each GPU set,
// so it is more convenient than just a nil pointer of type *Array.
// See: Alloc()
func NilArray(components int, size3D []int) *Array {
	t := new(Array)
	t.Init(components, size3D, DONT_ALLOC)
	return t
}

// If the array has no underlying storage yet (e.g., it was
// created by NilArray()), allocate that storage.
func (a *Array) Alloc() {
	Assert(a.pointer == 0)
	a.pointer = cu.MemAlloc(SIZEOF_FLOAT * int64(a.partLen4D))
	a.Zero()
	a.initCompPtrs() // need to update the component pointers
}

// Frees the underlying storage and sets the size to zero.
func (v *Array) Free() {
	v.Stream.Destroy()

	if v.pointer != 0 {
		v.pointer.Free()
		v.pointer = 0
	}

	for i := range v._size {
		v._size[i] = 0
	}
	v.initCompPtrs() // also set component pointers to NULL
	for c := range v.Comp {
		v.Comp[c].Stream.Destroy()
	}
}

// Address of part of the array on each GPU device
func (a *Array) DevicePtr() cu.DevicePtr {
	return a.pointer
}

// Total number of elements
func (a *Array) Len() int {
	return a._size[0] * a._size[1] * a._size[2] * a._size[3]
}

// Total number of elements per GPU
func (a *Array) PartLen4D() int {
	return a.partLen4D
}

// Number of elements per component per GPU
func (a *Array) PartLen3D() int {
	return a.partLen3D
}

// Number of components (1: scalar, 3: vector, ...).
func (a *Array) NComp() int {
	return a._size[0]
}

// Gets the i'th component as an array.
// E.g.: Component(0) is the x-component.
func (a *Array) Component(i int) *Array {
	if a._size[0] == 1 { // 1-component
		return a
	}
	return &(a.Comp[i])
}

// Array of pointers to parts, one per GPU.
func (a *Array) Pointer() cu.DevicePtr {
	return a.pointer
}

// True if the array has no underlying GPU storage.
// E.g., when created by NilArray()
func (a *Array) IsNil() bool {
	return a.pointer == 0
}

// Size of the vector field.
func (a *Array) Size3D() []int {
	return a.size3D
}

// Number of components + size of the vector field.
func (a *Array) Size4D() []int {
	return a.size4D
}

// Size of each part per GPU
func (a *Array) PartSize() []int {
	return a.partSize
}

// check if comp, x, y, z is inside the array's bounds.
// panic if not.
func (a *Array) checkBounds(comp, x, y, z int) {
	if comp < 0 || comp >= a.NComp() ||
		x < 0 || x >= a.size3D[X] ||
		y < 0 || y >= a.size3D[Y] ||
		z < 0 || z >= a.size3D[Z] {
		panic(InputErr(fmt.Sprint("gpu.Array index out of range. ",
			"component:", comp, " index:", z, y, x,
			" array size: ", a.NComp(), " components x ", a.size3D[Z], a.size3D[Y], a.size3D[X])))
	}
}

// Get a single value
func (b *Array) Get(comp, x, y, z int) float32 {
	b.checkBounds(comp, x, y, z)
	var value float32
	acomp := b.Comp[comp]
	index := acomp.indexOf(x, y, z)
	cu.MemcpyDtoH(cu.HostPtr(unsafe.Pointer(&value)),
		cu.DevicePtr(offset(uintptr(acomp.pointer), SIZEOF_FLOAT*index)),
		1*SIZEOF_FLOAT)
	return value
}

// Set a single value
func (b *Array) Set(comp, x, y, z int, value float32) {
	b.checkBounds(comp, x, y, z)
	acomp := b.Comp[comp]
	index := acomp.indexOf(x, y, z)
	cu.MemcpyHtoD(cu.DevicePtr(offset(uintptr(acomp.pointer), SIZEOF_FLOAT*index)),
		cu.HostPtr(unsafe.Pointer(&value)),
		1*SIZEOF_FLOAT)
}

func (a *Array) indexOf(x, y, z int) (index int) {
	N1 := a.partSize[Y]
	N2 := a.partSize[Z]
	index = x*N1*N2 + y*N2 + z
	return
}

// Copy from device array to device array.
func (dst *Array) CopyFromDevice(src *Array) {
	CheckSize(dst.size4D, src.size4D)

	d := dst.pointer
	s := src.pointer
	// copies run concurrently on the individual devices
	length := src.partLen4D
	cu.MemcpyDtoDAsync(cu.DevicePtr(d), cu.DevicePtr(s), SIZEOF_FLOAT*int64(length), cu.Stream(dst.Stream))
	// Synchronize with all copies
	dst.Stream.Sync()

}

// Copy from host array to device array.
func (dst *Array) CopyFromHost(src *host.Array) {
	CheckSize(dst.size4D, src.Size4D)

	partPlaneN := dst.partSize[1] * dst.partSize[2]    // floats per YZ plane per GPU
	planeN := dst.size3D[1] * dst.size3D[2]            // total floats per YZ plane
	NPlane := dst.size4D[0] * dst.size3D[0]            // total YZ planes (NComp * X size)
	partPlaneBytes := SIZEOF_FLOAT * int64(partPlaneN) // bytes per YZ plane per GPU

	for i := 0; i < NPlane; i++ {
		dstOffset := i * partPlaneN
		dstPtr := ArrayOffset(uintptr(dst.pointer), dstOffset)

		srcOffset := i * planeN

		cu.MemcpyHtoD(cu.DevicePtr(dstPtr), cu.HostPtr(&src.List[srcOffset]), partPlaneBytes)
	}
}

// Copy from device array to host array.
func (src *Array) CopyToHost(dst *host.Array) {
	CheckSize(dst.Size4D, src.size4D)

	partPlaneN := src.partSize[1] * src.partSize[2]    // floats per YZ plane per GPU
	planeN := src.size3D[1] * src.size3D[2]            // total floats per YZ plane
	NPlane := src.size4D[0] * src.size3D[0]            // total YZ planes (NComp * X size)
	partPlaneBytes := SIZEOF_FLOAT * int64(partPlaneN) // bytes per YZ plane per GPU

	for i := 0; i < NPlane; i++ {
		srcOffset := i * partPlaneN
		srcPtr := ArrayOffset(uintptr(src.pointer), srcOffset)

		dstOffset := i * planeN

		cu.MemcpyDtoH(cu.HostPtr(&dst.List[dstOffset]), cu.DevicePtr(srcPtr), partPlaneBytes)
	}

}

// DEBUG: Make a freshly allocated copy on the host.
func (src *Array) LocalCopy() *host.Array {
	dst := host.NewArray(src.NComp(), src.Size3D())
	src.CopyToHost(dst)
	return dst
}

// Makes all elements zero.
func (a *Array) Zero() {
	a.MemSet(0)
}

func (a *Array) MemSet(num float32) {
	slices := a.pointer
	cu.MemsetD32Async(slices, math.Float32bits(num), int64(a.partLen4D), cu.Stream(a.Stream))
	a.Stream.Sync()
}

// Error message.
const MSG_ARRAY_SIZE_MISMATCH = "array size mismatch"

// Pointer arithmetic: returns ptr + bytes.
// When ptr is NULL, NULL is returned.
func offset(ptr uintptr, bytes int) uintptr {
	if ptr == 0 {
		return 0
		//panic(Bug("offsetting null pointer"))
	}
	return ptr + uintptr(bytes)
}

// Human-readable string.
func (a *Array) String() string {
	return fmt.Sprint("gpu.Array{pointers=", a.pointer, "size=", a.size4D, "}")
}
