// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package runtime

// This file implements device management.

//#include <cuda_runtime.h>
import "C"

import (
	"fmt"
	"unsafe"
)

// Returns the device number on which the host thread executes device code.
func GetDevice() int {
	var device int
	err := Error(C.cudaGetDevice((*C.int)(unsafe.Pointer(&device))))
	if err != Success {
		panic(err)
	}
	return device
}

// Returns the number of GPUs
func GetDeviceCount() int {
	var count int
	err := Error(C.cudaGetDeviceCount((*C.int)(unsafe.Pointer(&count))))
	if err != Success {
		panic(err)
	}
	return count
}

// Sets the device on which the host thread executes device code.
func SetDevice(device int) {
	err := Error(C.cudaSetDevice(C.int(device)))
	if err != Success {
		panic(err)
	}
}

// Set device flags for the device associated with the host thread
func SetDeviceFlags(flags DeviceFlag) {
	err := Error(C.cudaSetDeviceFlags(C.uint(flags)))
	if err != Success {
		panic(err)
	}
}

type DeviceFlag uint

const (
	DeviceScheduleAuto    DeviceFlag = 0  //Automatic scheduling
	DeviceScheduleSpin    DeviceFlag = 1  //Spin default scheduling
	DeviceScheduleYield   DeviceFlag = 2  //Yield default scheduling
	DeviceBlockingSync    DeviceFlag = 4  //Use blocking synchronization
	DeviceMapHost         DeviceFlag = 8  //Support mapped pinned allocations
	DeviceLmemResizeToMax DeviceFlag = 16 //Keep local memory allocation after launch
)

// Returns the device properties
func GetDeviceProperties(device int) *DeviceProp {
	var prop C.struct_cudaDeviceProp
	err := Error(C.cudaGetDeviceProperties(&prop, C.int(device)))
	if err != Success {
		panic(err)
	}
	devProp := new(DeviceProp)

	devProp.Name = C.GoString(&prop.name[0])
	devProp.TotalGlobalMem = uint(prop.totalGlobalMem)
	devProp.SharedMemPerBlock = uint(prop.sharedMemPerBlock)
	devProp.RegsPerBlock = int(prop.regsPerBlock)
	devProp.WarpSize = int(prop.warpSize)
	devProp.MemPitch = int(prop.memPitch)
	devProp.MaxThreadsPerBlock = int(prop.maxThreadsPerBlock)
	devProp.MaxThreadsDim[0] = int(prop.maxThreadsDim[0])
	devProp.MaxThreadsDim[1] = int(prop.maxThreadsDim[1])
	devProp.MaxThreadsDim[2] = int(prop.maxThreadsDim[2])
	devProp.MaxGridSize[0] = int(prop.maxGridSize[0])
	devProp.MaxGridSize[1] = int(prop.maxGridSize[1])
	devProp.MaxGridSize[2] = int(prop.maxGridSize[2])
	devProp.TotalConstMem = uint(prop.totalConstMem)
	devProp.Major = int(prop.major)
	devProp.Minor = int(prop.minor)
	devProp.ClockRate = int(prop.clockRate)
	devProp.TextureAlignment = int(prop.textureAlignment)
	devProp.DeviceOverlap = int(prop.deviceOverlap)
	devProp.MultiProcessorCount = int(prop.multiProcessorCount)
	devProp.KernelExecTimeoutEnabled = int(prop.kernelExecTimeoutEnabled)
	devProp.Integrated = int(prop.integrated)
	devProp.CanMapHostMemory = int(prop.canMapHostMemory)
	devProp.ComputeMode = int(prop.computeMode)
	devProp.ConcurrentKernels = int(prop.concurrentKernels)
	devProp.ECCEnabled = int(prop.ECCEnabled)
	devProp.PciBusID = int(prop.pciBusID)
	devProp.PciDeviceID = int(prop.pciDeviceID)
	devProp.TccDriver = int(prop.tccDriver)
	return devProp
}

// Stores device properties
type DeviceProp struct {
	Name                     string
	TotalGlobalMem           uint
	SharedMemPerBlock        uint
	RegsPerBlock             int
	WarpSize                 int
	MemPitch                 int
	MaxThreadsPerBlock       int
	MaxThreadsDim            [3]int
	MaxGridSize              [3]int
	TotalConstMem            uint
	Major                    int
	Minor                    int
	ClockRate                int
	TextureAlignment         int
	DeviceOverlap            int
	MultiProcessorCount      int
	KernelExecTimeoutEnabled int
	Integrated               int
	CanMapHostMemory         int
	ComputeMode              int
	ConcurrentKernels        int
	ECCEnabled               int
	PciBusID                 int
	PciDeviceID              int
	TccDriver                int
}

func (prop *DeviceProp) String() string {
	return fmt.Sprintln("Name:                    ", prop.Name) +
		fmt.Sprintln("TotalGlobalMem:          ", prop.TotalGlobalMem) +
		fmt.Sprintln("SharedMemPerBlock:       ", prop.SharedMemPerBlock) +
		fmt.Sprintln("RegsPerBlock:            ", prop.RegsPerBlock) +
		fmt.Sprintln("WarpSize:                ", prop.WarpSize) +
		fmt.Sprintln("MemPitch:                ", prop.MemPitch) +
		fmt.Sprintln("MaxThreadsPerBlock:      ", prop.MaxThreadsPerBlock) +
		fmt.Sprintln("MaxThreadsDim:           ", prop.MaxThreadsDim) +
		fmt.Sprintln("MaxGridSize:             ", prop.MaxGridSize) +
		fmt.Sprintln("TotalConstMem:           ", prop.TotalConstMem) +
		fmt.Sprintln("Major:                   ", prop.Major) +
		fmt.Sprintln("Minor:                   ", prop.Minor) +
		fmt.Sprintln("ClockRate:               ", prop.ClockRate) +
		fmt.Sprintln("TextureAlignment:        ", prop.TextureAlignment) +
		fmt.Sprintln("DeviceOverlap:           ", prop.DeviceOverlap) +
		fmt.Sprintln("MultiProcessorCount:     ", prop.MultiProcessorCount) +
		fmt.Sprintln("KernelExecTimeoutEnabled:", prop.KernelExecTimeoutEnabled) +
		fmt.Sprintln("Integrated:              ", prop.Integrated) +
		fmt.Sprintln("CanMapHostMemory:        ", prop.CanMapHostMemory) +
		fmt.Sprintln("ComputeMode:             ", prop.ComputeMode) +
		fmt.Sprintln("ConcurrentKernels:       ", prop.ConcurrentKernels) +
		fmt.Sprintln("ECCEnabled:              ", prop.ECCEnabled) +
		fmt.Sprintln("PciBusID:                ", prop.PciBusID) +
		fmt.Sprintln("PciDeviceID:             ", prop.PciDeviceID) +
		fmt.Sprint("TccDriver:                ", prop.TccDriver)
}

type CacheConfig uint32

// L1/shared memory configuration flags
const (
	FuncCachePreferNone   CacheConfig = C.cudaFuncCachePreferNone
	FuncCachePreferShared CacheConfig = C.cudaFuncCachePreferShared
	FuncCachePreferL1     CacheConfig = C.cudaFuncCachePreferL1
	FuncCachePreferEqual  CacheConfig = C.cudaFuncCachePreferEqual
)

func DeviceSetCacheConfig(cconfig CacheConfig) {
	err := Error(C.cudaDeviceSetCacheConfig(uint32(cconfig)))
	if err != Success {
		panic(err)
	}
	return
}
