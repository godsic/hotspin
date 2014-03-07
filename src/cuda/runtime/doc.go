// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

//CUDA runtime bindings for Go.
//--------------------
//
//This package provides Go bindings for nVIDIA CUDA.
//Access is provided to the low-level functionality,
//together with high-level (more idiomatic Go) functions.
//
//Low-level usage example:
//
//	import "cuda"
//
//	runtime.LockOSThread()              // cuda contexts are tied to OS threads
//	pointer := cuda.Malloc(4 * 1024)    // low-level memory allocation
//	cuda.Free(pointer)                  // low level free
//
//
//High-level usage example:
//
//	import "cuda"
//
//	runtime.LockOSThread()                   // cuda contexts are tied to OS threads
//	deviceArray := cuda.NewArray(1024, 4)    // space for 1024 float32's
//	hostArray := make([]float32, 1024)
//	cuda.Copy(deviceArray, hostArray)        // copy from host to device
//	cuda.Copy(hostArray, deviceArray)        // copy from device to host
//	deviceArray.Free()
//
//
//CUDA error statuses are checked automatically and passed to panic() when not successful.
//Also, return values are returned directly, and not via a pointer passed as function argument.
//E.g.:
//
//The C code:
//	void *ptr;
//	cudaError_t error = cudaMalloc(&ptr, 1024);
//	if(error != cudaSuccess){
//		fprintf(stderr, cudaErrorString(error))
//		abort()
//	}
//
//Becomes in Go:
//
//	ptr := cuda.Malloc(1024)
//
//
//Installation
//------------
//
//* To use this package, the CUDA SDK must be installed.
//  It has been tested with CUDA 3.2 on Linux. See:
//	http://developer.nvidia.com/object/cuda_3_2_downloads.html
//
//* Have a look at cuda/Makefile.
//  The Makefile assumes that the header files (cuda.h and friends) are located in:
//  	/usr/local/cuda/include.
//  and that the libraries (cudart.so and friends) are locted in:
//	/usr/local/cuda/lib  /usr/local/cuda/lib64
//  Check that this applicable for your system and change these paths if neccesarey.
//
//* Finally:
//	make install
package runtime
