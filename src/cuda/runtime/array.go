// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package runtime

// This file implements safe access to cuda memory.

import ()

type Array interface {
	Pointer() uintptr
	Bytes() int
}

type AnyArray struct {
	pointer uintptr
	bytes   int
}

// Returns the address of the underlying storage
func (a *AnyArray) Pointer() uintptr {
	return a.pointer
}

// Returns the size of the array in bytes.
func (a *AnyArray) Bytes() int {
	return a.bytes
}

// Frees the Array's underlying storage.
// After freeing the array points to nil to avoid dangling pointers.
// See also: Array.IsNil()
func (a *AnyArray) Free() {
	pointer := a.pointer
	a.pointer = uintptr(0)
	if pointer != uintptr(0) {
		Free(pointer)
	}
}

// IsNil() returns true when the array is either not initialized or has been freed.
func (a *AnyArray) IsNil() bool {
	return a.pointer == uintptr(0)
}

type Float32Array struct {
	AnyArray
}

func (a *Float32Array) Len() int {
	return a.bytes / SIZEOF_FLOAT32
}

// New array holding the specified number of float32 elements
func NewFloat32Array(elements int) *Float32Array {
	a := new(Float32Array)
	a.bytes = elements * SIZEOF_FLOAT32
	a.pointer = Malloc(a.bytes)
	return a
}

type Complex64Array struct {
	AnyArray
}

func (a *Complex64Array) Len() int {
	return a.bytes / SIZEOF_COMPLEX64
}

// New array holding the specified number of float32 elements
func NewComplex64Array(elements int) *Complex64Array {
	a := new(Complex64Array)
	a.bytes = elements * SIZEOF_COMPLEX64
	a.pointer = Malloc(a.bytes)
	return a
}

//// New array holding the specified number of complex32 elements
//func NewComplex64Array(elements int) *Array {
//	return NewArray(elements, SIZEOF_COMPLEX64)
//}
//
//// New array holding the specified number of float32 elements
//func NewInt32Array(elements int) *Array {
//	return NewArray(elements, SIZEOF_INT32)
//
//}
//
//// New array holding the specified number of float64 elements
//func NewFloat64Array(elements int) *Array {
//	return NewArray(elements, SIZEOF_FLOAT64)
//}
//
//// New array holding the specified number of complex32 elements
//func NewComplex128Array(elements int) *Array {
//	return NewArray(elements, SIZEOF_COMPLEX128)
//}
//
//// New array holding the specified number of float64 elements
//func NewInt64Array(elements int) *Array {
//	return NewArray(elements, SIZEOF_INT64)
//}
//
//
//// Allocates a fresh CUDA array holding the specified number of elements with given size.
//// The array is initialized with zeros.
//func NewArray(bytes, size int) *Array {
//	a := new(Array)
//	a.init(bytes, size)
//	return a
//}
//
//
//
//// Refers to a raw CUDA device array
//type Array struct {
//	pointer  uintptr
//	bytes    int
//	elemsize int
//}
//
//
//// INTERNAL
//func (a *Array) init(bytes, elemsize int) {
//	Bytes := bytes * elemsize
//	a.pointer = Malloc(Bytes)
//	a.bytes = Bytes
//	a.elemsize = elemsize
//	Memset(a.pointer, 0, Bytes)
//}
//
//
//// Returns the address of the underlying storage
//func (a *Array) Pointer() uintptr {
//	return a.pointer
//}
//
//
//// Returns the size of the array in bytes.
//func (a *Array) Bytes() int {
//	return a.bytes
//}
//
//
//
//// INTERNAL. Function variant of the Free() method,
//// can be passed to runtime.SetFinalizer().
//func finalizeArray(a *Array) {
//	Free(a.pointer)
//}
