//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// This file implements GPU selection for multi-device operation.
// Author: Arne Vansteenkiste

//#include "libhotspin.h"
import "C"

import (
	cu "cuda/driver"
	cuda "cuda/runtime"
	. "hotspin-core/common"
)

// INTERNAL: List of GPU ids to use for multi-GPU operation. E.g.: {0,1,2,3}
var _useDevice int = 0

// INTERNAL: Device properties
var (
	maxThreadsPerBlock int
	maxBlockDim        [3]int
	maxGridDim         [3]int
)

// Sets a list of devices to use.
func InitGPU(device int, flags uint) {
	Debug("InitGPU ", device, flags)

	_useDevice = device
	cuda.SetDevice(_useDevice)

	printGPUInfo()
	initGPUProperties()
	initGPUL1Config()

	STREAM0 = NewStream()

}

// output device info
func printGPUInfo() {
	dev := cu.DeviceGet(_useDevice)
	Log("device", "( PCI", dev.Attribute(cu.PCI_DEVICE_ID), ")", dev.Name(), ",", dev.TotalMem()/(1024*1024), "MiB")
}

// set up device properties
func initGPUProperties() {
	dev := cu.DeviceGet(_useDevice)
	maxThreadsPerBlock = dev.Attribute(cu.MAX_THREADS_PER_BLOCK)
	maxBlockDim[0] = dev.Attribute(cu.MAX_BLOCK_DIM_X)
	maxBlockDim[1] = dev.Attribute(cu.MAX_BLOCK_DIM_Y)
	maxBlockDim[2] = dev.Attribute(cu.MAX_BLOCK_DIM_Z)
	maxGridDim[0] = dev.Attribute(cu.MAX_GRID_DIM_X)
	maxGridDim[1] = dev.Attribute(cu.MAX_GRID_DIM_Y)
	maxGridDim[2] = dev.Attribute(cu.MAX_GRID_DIM_Z)
	Debug("Max", maxThreadsPerBlock, "threads per block, max", maxGridDim, "x", maxBlockDim, "threads per GPU")
}

// init inter-device access
func initGPUL1Config() {
	// first init contexts
	setDevice(_useDevice)
	dummy := cuda.Malloc(1) // initializes a cuda context for the device
	cuda.Free(dummy)
	cuda.DeviceSetCacheConfig(cuda.FuncCachePreferL1)
}

// Error message
const ERR_UNIFIED_ADDR = "A GPU does not support unified addressing and can not be used in a multi-GPU setup."

// Assures Context ctx[id] is currently active. Switches contexts only when necessary.
func setDevice(deviceId int) {
	// actually set the device
	cuda.SetDevice(deviceId)
}

// Error message
const (
	MSG_BADDEVICEID       = "Invalid device ID: "
	MSG_DEVICEUNINITIATED = "Device list not initiated"
)

// Stream 0 on each GPU
var STREAM0 Stream
