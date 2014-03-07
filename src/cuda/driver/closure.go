// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package driver

// This file implements "closures", wrappers around a cuda function + argument list + launch configuration.
// The low-level func LaunchKernel() is rather arcane to use.

import (
	"fmt"
	"reflect"
	"unsafe"
)

// Wraps a CUDA Function, argument list, launch configuration and stream.
type Closure struct {
	Func           Function
	ArgVals        []uint64  // Values of arguments, stored in uint64 (large enough to hold any primitive value)
	ArgPtrs        []uintptr // Pointers to arguments, passed to launch
	BlockDim       [3]int
	GridDim        [3]int
	SharedMemBytes int
	Str            Stream
}

// Creates a closure for the function with argCount arguments.
func Close(f Function, argCount int) (c Closure) {
	c.Func = f
	c.ArgVals = make([]uint64, argCount)
	c.ArgPtrs = make([]uintptr, argCount)
	c.Str = STREAM0
	// sensible default config:
	for i := range c.BlockDim {
		c.BlockDim[i] = 1
		c.GridDim[i] = 1
	}
	return
}

// Sets the kernel launch configuration.
func (c *Closure) SetConfig(gridDim, blockDim []int) {
	for i := range c.BlockDim {
		c.BlockDim[i] = blockDim[i]
		c.GridDim[i] = gridDim[i]
	}
}

// Error message.
const UNKNOWN_ARG_TYPE = "Can not handle argument type %v %v"

// Sets an argument.
func (c *Closure) SetArg(argIndex int, value interface{}) {
	switch value.(type) {
	default:
		panic(fmt.Sprintf(UNKNOWN_ARG_TYPE, value, reflect.TypeOf(value).Kind()))
	case int:
		c.Seti(argIndex, value.(int))
	case float32:
		c.Setf(argIndex, value.(float32))
	case DevicePtr:
		c.SetDevicePtr(argIndex, value.(DevicePtr))
	}
}

// Sets an integer argument.
func (c *Closure) Seti(argIndex int, value int) {
	c.ArgPtrs[argIndex] = uintptr(unsafe.Pointer(&(c.ArgVals[argIndex])))
	*((*int)(unsafe.Pointer(c.ArgPtrs[argIndex]))) = value
}

// Sets a float32 argument.
func (c *Closure) Setf(argIndex int, value float32) {
	c.ArgPtrs[argIndex] = uintptr(unsafe.Pointer(&(c.ArgVals[argIndex])))
	*((*float32)(unsafe.Pointer(c.ArgPtrs[argIndex]))) = value
}

// Sets a device pointer argument.
func (c *Closure) SetDevicePtr(argIndex int, value DevicePtr) {
	c.ArgPtrs[argIndex] = uintptr(unsafe.Pointer(&(c.ArgVals[argIndex])))
	*((*DevicePtr)(unsafe.Pointer(c.ArgPtrs[argIndex]))) = value
}

// Executes the closure without waiting for the result.
// See: Closure.Synchronize()
func (c *Closure) Go() {
	LaunchKernel(c.Func, c.GridDim[0], c.GridDim[1], c.GridDim[2], c.BlockDim[0], c.BlockDim[1], c.BlockDim[2], c.SharedMemBytes, c.Str, c.ArgPtrs)
}

func (c *Closure) Synchronize() {
	c.Str.Synchronize()
}

func (c *Closure) Call() {
	c.Go()
	c.Synchronize()
}
