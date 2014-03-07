// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package driver

// This file implements CUDA driver initialization

//#include <cuda.h>
import "C"

import ()

// Initialize the CUDA driver API.
// Currently, flags must be 0.  
// If Init() has not been called, any function from the driver API will panic with ERROR_NOT_INITIALIZED.
func Init(flags int) {
	err := Result(C.cuInit(C.uint(flags)))
	if err != SUCCESS {
		panic(err)
	}
}
