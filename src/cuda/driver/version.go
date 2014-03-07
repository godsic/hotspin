// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package driver

// This file implements CUDA driver version management

//#include <cuda.h>
import "C"
import ()

// Returns the CUDA driver version.
func Version() int {
	var version C.int
	err := Result(C.cuDriverGetVersion(&version))
	if err != SUCCESS {
		panic(err)
	}
	return int(version)
}
