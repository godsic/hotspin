//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

// DO NOT USE TEST.FATAL: -> runtime.GoExit -> context switch -> INVALID CONTEXT!

package gpu

// Author: Arne Vansteenkiste

import (
	. "hotspin-core/common"
	"testing"
	//"fmt"
)

func TestCombineZ(test *testing.T) {
	// fail test on panic, do not crash
	defer func() {
		if err := recover(); err != nil {
			test.Error(err)
		}
	}()

	Nc := 3
	size1 := []int{2, 4, 8}
	size2 := []int{2, 4, 8 * 2}

	a := NewArray(Nc, size1)
	defer a.Free()
	ah := a.LocalCopy()

	for i := range ah.List {
		ah.List[i] = float64(i)
	}
	a.CopyFromHost(ah)

	b := NewArray(Nc, size1)
	defer b.Free()
	bh := a.LocalCopy()
	for i := range bh.List {
		bh.List[i] = -float64(i)
	}
	b.CopyFromHost(bh)

	c := NewArray(Nc, size2)
	defer c.Free()

	//fmt.Println("a", a.LocalCopy().Array)
	//fmt.Println("b", b.LocalCopy().Array)

	InsertBlockZ(c, a, 0)
	InsertBlockZ(c, b, 1)

	//fmt.Println("c", c.LocalCopy().Array)

	A := a.LocalCopy().Array
	B := b.LocalCopy().Array
	S0, S1, S2 := ah.Size3D[0], ah.Size3D[1], ah.Size3D[2]
	C := c.LocalCopy().Array
	for cmp := range C {
		for i := range C[cmp] {
			for j := range C[cmp][i] {
				for k := range C[cmp][i][j] {
					if i < S0 && j < S1 && k < S2 {
						if A[cmp][i][j][k] != C[cmp][i][j][k] {
							test.Fail()
						}
					} else {
						if B[cmp][i][j][k-S2] != C[cmp][i][j][k] {
							test.Fail()
						}
					}
				}
			}
		}
	}
}

func TestCopyPadZ(test *testing.T) {
	// fail test on panic, do not crash
	defer func() {
		if err := recover(); err != nil {
			test.Error(err)
		}
	}()

	size1 := []int{2, 4, 8}
	size2 := []int{2, 4, 8 + 2}

	a := NewArray(3, size1)
	defer a.Free()
	ah := a.LocalCopy()

	b := NewArray(3, size2)
	b.MemSet(42)
	defer b.Free()

	for i := range ah.List {
		ah.List[i] = float64(i)
	}

	a.CopyFromHost(ah)

	//fmt.Println("CopyPadZ", b.LocalCopy())
	CopyPadZ(b, a)
	bh := b.LocalCopy()
	//	fmt.Println("CopyPadZ", bh.Array)

	A := ah.Array
	S0, S1, S2 := ah.Size3D[0], ah.Size3D[1], ah.Size3D[2]
	B := bh.Array
	for c := range B {
		for i := range B[c] {
			for j := range B[c][i] {
				for k := range B[c][i][j] {
					if i < S0 && j < S1 && k < S2 {
						if A[c][i][j][k] != B[c][i][j][k] {
							test.Fail()
						}
					} else {
						if B[c][i][j][k] != 0 {
							test.Fail()
						}
					}
				}
			}
		}
	}

	c := NewArray(3, size1)
	c.MemSet(42)
	CopyPadZ(c, b)
	//	fmt.Println("CopyPadZ", c.LocalCopy().Array)

	C := c.LocalCopy().Array
	for c := range B {
		for i := range B[c] {
			for j := range B[c][i] {
				for k := range B[c][i][j] {
					if i < S0 && j < S1 && k < S2 {
						if C[c][i][j][k] != B[c][i][j][k] {
							test.Fail()
						}
					} else {
						if B[c][i][j][k] != 0 {
							test.Fail()
						}
					}
				}
			}
		}
	}

}

func BenchmarkCopyPadZ(b *testing.B) {
	b.StopTimer()

	size := bigsize()
	a := NewArray(3, size)
	a2 := NewArray(3, []int{size[0], size[1], size[2] + 2})
	b.SetBytes(int64(a.Len()) * SIZEOF_FLOAT)
	defer a.Free()
	defer a2.Free()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		CopyPadZ(a2, a)
	}
}
