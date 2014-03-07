// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package runtime

// This file implements low-level kernel execution control,
// allowing kernels to be launced directly form go
// (equivalent to CUDA's <<< >>> syntax).
// See safe_launch.go for a safe version of this implemenation.

//#include <cuda_runtime.h>
import "C"
import "unsafe"

import ()

type dim3 struct {
	X, Y, Z uint
}

func Dim3(x, y, z uint) dim3 {
	return dim3{x, y, z}
}

func Dim(n uint) dim3 {
	return dim3{n, 1, 1}
}

// UNSAFE
// Push grid size on the execution stack. To be followed by SetupArgument()
func ConfigureCall(gridDim, blockDim dim3, sharedMem uint, stream Stream) {
	var grid, block C.dim3
	grid.x = C.uint(gridDim.X)
	grid.y = C.uint(gridDim.Y)
	grid.z = C.uint(gridDim.Z)
	block.x = C.uint(blockDim.X)
	block.y = C.uint(blockDim.Y)
	block.z = C.uint(blockDim.Z)
	err := Error(C.cudaConfigureCall((grid), (block), C.size_t(sharedMem), C.cudaStream_t(unsafe.Pointer(uintptr(stream)))))
	if err != Success {
		panic(err)
	}
}

// UNSAFE
// Push arguments on the execution stack. To be preceded by ConfigureCall()
func SetupArgument(arg uintptr, size, offset uint) {
	err := Error(C.cudaSetupArgument(unsafe.Pointer(arg), C.size_t(size), C.size_t(offset)))
	if err != Success {
		panic(err)
	}
}

// UNSAFE
// Launches the function on the device. To be preceded by ConfigureCall()
func Launch(entry string) {
	err := Error(C.cudaLaunch(unsafe.Pointer(C.CString(entry))))
	if err != Success {
		panic(err)
	}
}

//func FuncGetAttributes() FuncAttributes{
//
//}

type FuncAttributes struct {
	// Size (bytes) of statically allocated shared memory per block.
	SharedSizeBytes uint

	// Size (bytes) of user-allocated constant memory.
	ConstSizeBytes uint

	// Size (bytes) of local memory used by each thread.
	LocalSizeBytes uint

	// Maximum number of threads per block.
	MaxThreadsPerBlock int

	//Number of registers used by each thread.
	NumRegs int

	// PTX virtual architecture version. Major version * 10 + minor version.
	PtxVersion int

	// binary architecture version. Major version * 10 + minor version.
	BinaryVersion int
}
