// Copyright 2013 Mykola Dvornik (mykola.dvornik@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package curand

import "C"

//#cgo LDFLAGS: -L/usr/lib64/nvidia -L/usr/local/cuda/lib64/  -lcuda -lcudart -lcurand
//#cgo CFLAGS:  -I/usr/local/cuda/include/  -Wno-error
import "C"
