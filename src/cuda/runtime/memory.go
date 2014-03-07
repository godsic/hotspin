// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package runtime

// This file implements low-level memory management.

//#include <cuda_runtime.h>
import "C"
import "unsafe"

import ()


type HostRegisterType uint32

// Memory copy flag for Memcpy
const (
	HostRegisterPortable     HostRegisterType = C.cudaHostRegisterPortable
	HostRegisterMapped       HostRegisterType = C.cudaHostRegisterMapped
)

// Low-level page-locking of paged memory buffer.
// Must be explicitly unregistered with HostUnregister
func HostRegister(buffer uintptr, bytes int64, memtype HostRegisterType) {
	err := Error(C.cudaHostRegister((unsafe.Pointer)(buffer), C.size_t(bytes), C.uint(memtype)))
	if err != Success {
		panic(err)
	}
	return
}

// Low-level page-ulocking of locked memory buffer.
func HostUnregister(buffer uintptr) {
	err := Error(C.cudaHostUnregister((unsafe.Pointer)(buffer)))
	if err != Success {
		panic(err)
	}
	return
}


// Low-level memory allocation. Not memory-safe and not garbage collected.
// Must be freed with Free().
// NewArray() provides a memory-safe and garbage collected alternative.
func Malloc(bytes int) uintptr {
	var arr unsafe.Pointer
	err := Error(C.cudaMalloc((*unsafe.Pointer)(&arr), C.size_t(bytes)))
	if err != Success {
		panic(err)
	}
	return uintptr(arr)
}

// Low-level free of memory allocated by Malloc.
func Free(array uintptr) {
	err := Error(C.cudaFree(unsafe.Pointer(array)))
	if err != Success {
		panic(err)
	}
}

// Sets the first count bytes to value.
func Memset(devPtr uintptr, value int, count int) {
	err := Error(C.cudaMemset(unsafe.Pointer(devPtr), C.int(value), C.size_t(count)))
	if err != Success {
		panic(err)
	}
}

// Low-level unsafe memory copy
func Memcpy(dest, source uintptr, bytes int, direction MemcpyKind) {
	//println("Memcpy ", dest, ", ", source, ", ", bytes, ", ", direction)
	err := Error(C.cudaMemcpy(unsafe.Pointer(dest), unsafe.Pointer(source), C.size_t(bytes), uint32(direction)))
	if err != Success {
		panic(err)
	}
}

// Low-level asynchronous unsafe memory copy
// Works on device memory or page-locked host memory.
func MemcpyAsync(dest, source uintptr, bytes int, direction MemcpyKind, stream Stream) {
	err := Error(C.cudaMemcpyAsync(unsafe.Pointer(dest), unsafe.Pointer(source), C.size_t(bytes), uint32(direction), C.cudaStream_t(unsafe.Pointer(uintptr(stream)))))
	if err != Success {
		panic(err)
	}
}

type MemcpyKind uint32

// Memory copy flag for Memcpy
const (
	MemcpyHostToHost     MemcpyKind = C.cudaMemcpyHostToHost
	MemcpyHostToDevice   MemcpyKind = C.cudaMemcpyHostToDevice
	MemcpyDeviceToHost   MemcpyKind = C.cudaMemcpyDeviceToHost
	MemcpyDeviceToDevice MemcpyKind = C.cudaMemcpyDeviceToDevice
)

// Gets the address of the symbol on the device (variable in global or constant memory space)
func GetSymbolAddress(symbol string) uintptr {
	var devptr unsafe.Pointer
	err := Error(C.cudaGetSymbolAddress((*unsafe.Pointer)(&devptr), unsafe.Pointer(C.CString(symbol))))
	if err != Success {
		panic(err)
	}
	return uintptr(devptr)
}

// Gets the size (bytes) of the symbol on the device (variable in global or constant memory space)
func GetSymbolSize(symbol string) int {
	var size C.size_t
	err := Error(C.cudaGetSymbolSize(&size, unsafe.Pointer(C.CString(symbol))))
	if err != Success {
		panic(err)
	}
	return int(size)
}

const (
	SIZEOF_FLOAT32    = 4
	SIZEOF_COMPLEX64  = 8
	SIZEOF_INT32      = 4
	SIZEOF_FLOAT64    = 8
	SIZEOF_COMPLEX128 = 16
	SIZEOF_INT64      = 8
)
