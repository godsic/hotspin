//  This file is part of MuMax, a high-perfomrance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// Author: Arne Vansteenkiste

import (
	"fmt"
	. "hotspin-core/common"
	"hotspin-core/host"
	"runtime"
	"testing"
)

// Test repeated alloc/free.
func TestArrayAlloc(t *testing.T) {
	runtime.LockOSThread()

	N := BIG / 16
	size := []int{1, 4, N}
	for i := 0; i < 100; i++ {
		t := NewArray(3, size)
		t.Free()
	}
}

// Should init to zeros
func TestArrayInit(test *testing.T) {
	runtime.LockOSThread()
	// fail test on panic, do not crash
	//defer func() {
	//	if err := recover(); err != nil {
	//		test.Error(err)
	//	}
	//}()

	size := []int{4, 8, 16}
	host1 := host.NewArray(3, size)
	dev1 := NewArray(3, size)
	defer dev1.Free()
	fmt.Printf("%+v\n", dev1)

	if dev1.Len() != 3*Prod(size) {
		if !test.Failed() {
			test.Error("Len(): ", dev1.Len(), "expected: ", 3*Prod(size))
		}
	}

	if dev1.PartLen4D() != 3*Prod(size)/NDevice() {
		test.Fail()
	}

	if dev1.PartLen3D() != Prod(size)/NDevice() {
		test.Fail()
	}

	l1 := host1.List
	for i := range l1 {
		l1[i] = float64(i)
	}

	dev1.CopyToHost(host1)
	//host1.CopyFromDevice(dev1)

	for i := range l1 {
		if l1[i] != 0 {
			if !test.Failed() {
				test.Error(l1[i], "!=0")
			}
		}
	}
}

func TestArrayCopy(test *testing.T) {
	runtime.LockOSThread()
	// fail test on panic, do not crash
	//defer func() {
	//	if err := recover(); err != nil {
	//		test.Error(err)
	//	}
	//}()

	size := []int{4, 8, 16}
	host1, host2 := host.NewArray(3, size), host.NewArray(3, size)
	dev1, dev2 := NewArray(3, size), NewArray(3, size)
	defer dev1.Free()
	defer dev2.Free()

	l1 := host1.List
	for i := range l1 {
		l1[i] = float64(i)
	}

	dev1.CopyFromHost(host1)
	dev2.CopyFromDevice(dev1)
	dev2.CopyToHost(host2)

	l2 := host2.List
	for i := range l1 {
		if l2[i] != float64(i) {
			if !test.Failed() {
				test.Error("expected", i, "got:", l2[i])
			}
		}
	}
}

func TestArrayCopyHost(test *testing.T) {
	size := []int{2, 4, 8}
	a := NewArray(1, size)
	defer a.Free()

	ah := a.LocalCopy()
	for i := range ah.List {
		ah.List[i] = float64(i)
	}

	a.CopyFromHost(ah)

	raw := a.RawCopy()
	fmt.Println(raw)
}
