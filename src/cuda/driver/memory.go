// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package driver

// This file implements CUDA memory management on the driver level

//#include <cuda.h>
import "C"

import (
	"fmt"
	"unsafe"
)

type DevicePtr uintptr
type HostPtr unsafe.Pointer

// Allocates a number of bytes of device memory.
func MemAlloc(bytes int64) DevicePtr {
	var devptr C.CUdeviceptr
	err := Result(C.cuMemAlloc(&devptr, C.size_t(bytes)))
	if err != SUCCESS {
		panic(err)
	}
	return DevicePtr(devptr)
}

// Frees device memory allocated by MemAlloc().
// Overwrites the pointer with NULL.
// It is safe to double-free.
func MemFree(ptr *DevicePtr) {
	p := *ptr
	if p == DevicePtr(uintptr(0)) {
		return // Allready freed
	}
	*ptr = DevicePtr(uintptr(0))
	err := Result(C.cuMemFree(C.CUdeviceptr(p)))
	if err != SUCCESS {
		panic(err)
	}
}

// Frees device memory allocated by MemAlloc().
// Overwrites the pointer with NULL.
// It is safe to double-free.
func (ptr *DevicePtr) Free() {
	MemFree(ptr)
}

// Copies a number of bytes on the current device.
// Requires unified addressing to be supported.
// See also: MemcpyDtoD().
// TODO(a): is actually an auto copy for device and/or host memory
func Memcpy(dst, src DevicePtr, bytes int64) {
	err := Result(C.cuMemcpy(C.CUdeviceptr(dst), C.CUdeviceptr(src), C.size_t(bytes)))
	if err != SUCCESS {
		panic(err)
	}
}

// Asynchronously copies a number of bytes on the current device.
func MemcpyAsync(dst, src DevicePtr, bytes int64, stream Stream) {
	err := Result(C.cuMemcpyAsync(C.CUdeviceptr(dst), C.CUdeviceptr(src), C.size_t(bytes), C.CUstream(unsafe.Pointer(uintptr(stream)))))
	if err != SUCCESS {
		panic(err)
	}
}

// Copies a number of bytes from host to device.
func MemcpyDtoD(dst, src DevicePtr, bytes int64) {
	err := Result(C.cuMemcpyDtoD(C.CUdeviceptr(dst), C.CUdeviceptr(src), C.size_t(bytes)))
	if err != SUCCESS {
		panic(err)
	}
}

// Asynchronously copies a number of bytes from host to device.
func MemcpyDtoDAsync(dst, src DevicePtr, bytes int64, stream Stream) {
	err := Result(C.cuMemcpyDtoDAsync(C.CUdeviceptr(dst), C.CUdeviceptr(src), C.size_t(bytes), C.CUstream(unsafe.Pointer(uintptr(stream)))))
	if err != SUCCESS {
		panic(err)
	}
}

// Copies a number of bytes from host to device.
func MemcpyHtoD(dst DevicePtr, src HostPtr, bytes int64) {
	err := Result(C.cuMemcpyHtoD(C.CUdeviceptr(dst), unsafe.Pointer(src), C.size_t(bytes)))
	if err != SUCCESS {
		panic(err)
	}
}

// Asynchronously copies a number of bytes from host to device.
// The host memory must be page-locked (see MemRegister)
func MemcpyHtoDAsync(dst DevicePtr, src HostPtr, bytes int64, stream Stream) {
	err := Result(C.cuMemcpyHtoDAsync(C.CUdeviceptr(dst), unsafe.Pointer(src), C.size_t(bytes), C.CUstream(unsafe.Pointer(uintptr(stream)))))
	if err != SUCCESS {
		panic(err)
	}
}

// Copies a number of bytes from device to host.
func MemcpyDtoH(dst HostPtr, src DevicePtr, bytes int64) {
	err := Result(C.cuMemcpyDtoH(unsafe.Pointer(dst), C.CUdeviceptr(src), C.size_t(bytes)))
	if err != SUCCESS {
		panic(err)
	}
}

// Asynchronously copies a number of bytes device host to host.
// The host memory must be page-locked (see MemRegister)
func MemcpyDtoHAsync(dst HostPtr, src DevicePtr, bytes int64, stream Stream) {
	err := Result(C.cuMemcpyDtoHAsync(unsafe.Pointer(uintptr(dst)), C.CUdeviceptr(src), C.size_t(bytes), C.CUstream(unsafe.Pointer(uintptr(stream)))))
	if err != SUCCESS {
		panic(err)
	}
}

// Copies from device memory in one context (device) to another.
func MemcpyPeer(dst DevicePtr, dstCtx Context, src DevicePtr, srcCtx Context, bytes int64) {
	err := Result(C.cuMemcpyPeer(C.CUdeviceptr(dst), C.CUcontext(unsafe.Pointer(uintptr(dstCtx))), C.CUdeviceptr(src), C.CUcontext(unsafe.Pointer(uintptr(srcCtx))), C.size_t(bytes)))
	if err != SUCCESS {
		panic(err)
	}
}

// Asynchronously copies from device memory in one context (device) to another.
func MemcpyPeerAsync(dst DevicePtr, dstCtx Context, src DevicePtr, srcCtx Context, bytes int64, stream Stream) {
	err := Result(C.cuMemcpyPeerAsync(C.CUdeviceptr(dst), C.CUcontext(unsafe.Pointer(uintptr(dstCtx))), C.CUdeviceptr(src), C.CUcontext(unsafe.Pointer(uintptr(srcCtx))), C.size_t(bytes), C.CUstream(unsafe.Pointer(uintptr(stream)))))
	if err != SUCCESS {
		panic(err)
	}
}

// Returns the base address and size of the allocation (by MemAlloc) that contains the input pointer ptr.
func MemGetAddressRange(ptr DevicePtr) (bytes int64, base DevicePtr) {
	var cbytes C.size_t
	var cptr C.CUdeviceptr
	err := Result(C.cuMemGetAddressRange(&cptr, &cbytes, C.CUdeviceptr(ptr)))
	if err != SUCCESS {
		panic(err)
	}
	bytes = int64(cbytes)
	base = DevicePtr(cptr)
	return
}

// Returns the base address and size of the allocation (by MemAlloc) that contains the input pointer ptr.
func (ptr DevicePtr) GetAddressRange() (bytes int64, base DevicePtr) {
	return MemGetAddressRange(ptr)
}

// Returns the size of the allocation (by MemAlloc) that contains the input pointer ptr.
func (ptr DevicePtr) Bytes() (bytes int64) {
	bytes, _ = MemGetAddressRange(ptr)
	return
}

// Returns the free and total amount of memroy in the current Context (in bytes).
func MemGetInfo() (free, total int64) {
	var cfree, ctotal C.size_t
	err := Result(C.cuMemGetInfo(&cfree, &ctotal))
	if err != SUCCESS {
		panic(err)
	}
	free = int64(cfree)
	total = int64(ctotal)
	return
}

// Page-locks memory specified by the pointer and bytes.
// The pointer and byte size must be aligned to the host page size (4KB)
// See also: MemHostUnregister()
func MemHostRegister(ptr HostPtr, bytes int64, flags MemHostRegisterFlag) {
	err := Result(C.cuMemHostRegister(unsafe.Pointer(ptr), C.size_t(bytes), C.uint(flags)))
	if err != SUCCESS {
		panic(err)
	}
}

// Unmaps memory locked by MemHostRegister().
func MemHostUnregister(ptr HostPtr) {
	err := Result(C.cuMemHostUnregister(unsafe.Pointer(uintptr(ptr))))
	if err != SUCCESS {
		panic(err)
	}
}

type MemHostRegisterFlag uint

// Flag for MemHostRegister
const (
	// Memory is pinned in all CUDA contexts.
	MEMHOSTREGISTER_PORTABLE MemHostRegisterFlag = C.CU_MEMHOSTREGISTER_PORTABLE
	// Maps the allocation in CUDA address space. TODO(a): cuMemHostGetDevicePointer()
	MEMHOSTREGISTER_DEVICEMAP MemHostRegisterFlag = C.CU_MEMHOSTREGISTER_DEVICEMAP
)

func (p DevicePtr) String() string {
	return fmt.Sprint(unsafe.Pointer(uintptr(p)))
}
