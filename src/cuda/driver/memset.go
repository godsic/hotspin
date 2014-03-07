// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package driver

// This file implements CUDA memset functions.

//#include <cuda.h>
import "C"

import (
	"unsafe"
)

// Sets the first N 32-bit values of dst array to value.
func MemsetD32(dst DevicePtr, value uint32, N int64) {
	err := Result(C.cuMemsetD32(C.CUdeviceptr(dst), C.uint(value), C.size_t(N)))
	if err != SUCCESS {
		panic(err)
	}
}

// Asynchronously sets the first N 32-bit values of dst array to value.
func MemsetD32Async(dst DevicePtr, value uint32, N int64, stream Stream) {
	err := Result(C.cuMemsetD32Async(C.CUdeviceptr(dst), C.uint(value), C.size_t(N), C.CUstream(unsafe.Pointer(uintptr(stream)))))
	if err != SUCCESS {
		panic(err)
	}
}
