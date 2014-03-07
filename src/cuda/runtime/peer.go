// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package runtime

// This file implements device peer management.

//#include <cuda_runtime.h>
import "C"

import ()

// Returns whether peer access is possible between device and peerDevice.
func DeviceCanAccessPeer(device, peerDevice int) bool {
	var can C.int
	err := Error(C.cudaDeviceCanAccessPeer(&can, C.int(device), C.int(peerDevice)))
	if err != Success {
		panic(err)
	}
	return int(can) != 0

}

// Enables memory on peerDevice to be accessed by the current device.
// The call is unidirectional and a symmetric call is needed for access
// in the opposite direction.
func DeviceEnablePeerAccess(peerDevice int) {
	err := Error(C.cudaDeviceEnablePeerAccess(C.int(peerDevice), 0))
	if err != Success {
		panic(err)
	}
}

// Disables memory on peerDevice to be accessed by the current device.
func DeviceDisablePeerAccess(peerDevice int) {
	err := Error(C.cudaDeviceDisablePeerAccess(C.int(peerDevice)))
	if err != Success {
		panic(err)
	}
}
