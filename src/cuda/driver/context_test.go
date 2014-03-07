// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package driver

import (
	"fmt"
	"testing"
)

func TestContext(t *testing.T) {
	fmt.Println("CtxCreate")
	ctx := CtxCreate(CTX_SCHED_AUTO, 0)
	fmt.Println("CtxSetCurrent")
	CtxSetCurrent(ctx)
	fmt.Println("CtxGetApiVersion:", ctx.ApiVersion())
	fmt.Println("CtxGetDevice:", CtxGetDevice())
	(&ctx).Destroy()
}
