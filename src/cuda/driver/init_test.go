// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package driver

import (
	"fmt"
)

// needed for all other tests.
func init() {
	Init(0)
	ctx := CtxCreate(CTX_SCHED_AUTO, 0)
	CtxSetCurrent(ctx)
	fmt.Println("Created CUDA context")
}
