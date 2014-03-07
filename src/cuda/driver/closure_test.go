// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package driver

import (
	"fmt"
	"testing"
)

func TestClosure(t *testing.T) {
	defer func() {
		err := recover()
		if err != nil {
			fmt.Println(err)
			t.Fail()
		}
	}()

	mod := ModuleLoad("testmodule.ptx")
	c := Close(mod.GetFunction("testMemset"), 3)

	N := 1024
	N4 := 4 * int64(N)
	a := make([]float32, N)

	A := MemAlloc(N4)
	aptr := HostPtr(&a[0])
	MemcpyHtoD(A, aptr, N4)

	c.SetDevicePtr(0, A)
	c.SetArg(1, float32(42.))
	c.Seti(2, N/2)

	c.BlockDim[0] = 128
	c.GridDim[0] = DivUp(N, 128)
	c.Call()

	MemcpyDtoH(aptr, A, N4)
	for i := 0; i < N/2; i++ {
		if a[i] != 42 {
			t.Fail()
		}
	}
	for i := N / 2; i < N; i++ {
		if a[i] != 0 {
			t.Fail()
		}
	}
	//fmt.Println(a)
}
