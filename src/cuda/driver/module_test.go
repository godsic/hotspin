// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package driver

import (
	"testing"
	"unsafe"
)

func TestModule(test *testing.T) {
	mod := ModuleLoad("testmodule.ptx")
	f := mod.GetFunction("testMemset")

	N := 1000
	N4 := 4 * int64(N)
	a := make([]float32, N)
	A := MemAlloc(N4)
	aptr := HostPtr(&a[0])
	MemcpyHtoD(A, aptr, N4)

	var array uintptr
	array = uintptr(A)

	var value float32
	value = 42

	var n int
	n = N / 2

	args := []uintptr{(uintptr)(unsafe.Pointer(&array)), (uintptr)(unsafe.Pointer(&value)), (uintptr)(unsafe.Pointer(&n))}
	block := 128
	grid := DivUp(N, block)
	shmem := 0
	stream := STREAM0
	LaunchKernel(f, grid, 1, 1, block, 1, 1, shmem, stream, args)

	MemcpyDtoH(aptr, A, N4)
	for i := 0; i < N/2; i++ {
		if a[i] != 42 {
			test.Fail()
		}
	}
	for i := N / 2; i < N; i++ {
		if a[i] != 0 {
			test.Fail()
		}
	}
	//fmt.Println(a)
}

// Integer division rounded up.
func DivUp(x, y int) int {
	return ((x - 1) / y) + 1
}
