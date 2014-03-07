// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package runtime

// This file implements safe(r) kernel execution control,
// allowing kernels to be launced directly form go
// (equivalent to CUDA's <<< >>> syntax).
// See execution.go for unsafe, low-level functions.

//#include <cuda_runtime.h>
import "C"

import (
	"reflect"
)

func CallAsync(function string, config *LaunchConfig, arguments ...interface{}) {
	ConfigureCall(config.GridDim, config.BlockDim, config.SharedMem, config.Stream) // TODO: inline
	for _, arg := range arguments {
		argvalue := reflect.ValueOf(arg)
		addr := argvalue.UnsafeAddr()
		size := uint(argvalue.Type().Size())
		SetupArgument(addr, size, 0)
	}
	Launch(function)
}

type LaunchConfig struct {
	GridDim   dim3
	BlockDim  dim3
	SharedMem uint
	Stream
}
