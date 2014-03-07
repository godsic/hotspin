// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package runtime

// This file implements synchronization with the host

//#include <cuda_runtime.h>
import "C"

import ()

// Blocks until all tasks have finished.
// Returns an error if a preceding task has failed.
func ThreadSynchronize() {
	err := Error(C.cudaThreadSynchronize())
	if err != Success {
		panic(err)
	}
}

// Explicitly clean up all CUDA resources.
func ThreadExit() {
	err := Error(C.cudaThreadExit())
	if err != Success {
		panic(err)
	}
}
