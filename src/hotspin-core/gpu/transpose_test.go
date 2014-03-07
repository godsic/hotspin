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
//"testing"
//"fmt"
)

//func TestTranspose(test *testing.T) {
//
//	N1 := 8
//	N2 := 16
//	size1 := []int{1, N1, N2 * 2}
//	size2 := []int{1, N2, N1 * 2}
//
//	const nComp = 1
//	a := NewArray(nComp, size1)
//	defer a.Free()
//	ah := a.LocalCopy()
//
//	b := NewArray(nComp, size2)
//	//b.MemSet(42)
//	defer b.Free()
//
//	for i := range ah.List {
//		ah.List[i] = float32(i)
//	}
//
//	a.CopyFromHost(ah)
//
//	fmt.Println("A", a.LocalCopy().List)
//	TransposeComplexYZ(b, a)
//	bh := b.LocalCopy()
//	fmt.Println("B", bh.List)
//
//	A := ah.Array
//	B := bh.Array
//	for c := range B {
//		for i := range B[c] {
//			for j := range B[c][i] {
//				for k := 0; k < len(B[c][i][j])/2; k++ {
//					if A[c][i][k][2*j] != B[c][i][j][2*k] ||
//						A[c][i][k][2*j+1] != B[c][i][j][2*k+1] {
//						test.Error(A[c][i][k][2*j+0], A[c][i][k][2*j+1], "!=", B[c][i][j][2*k+0], B[c][i][j][2*k+1])
//					}
//				}
//			}
//		}
//	}
//}

//func TestTransposePart(test *testing.T) {
//
//	if NDevice() > 1 {
//		println("Skipping TestTransposePart on >1 GPU")
//		return
//	}
//
//	size1 := []int{2, 20, 80 * 2}
//	size2 := []int{2, 80, 20 * 2}
//
//	a := NewArray(3, size1)
//	defer a.Free()
//	ah := a.LocalCopy()
//
//	b := NewArray(3, size2)
//	b.MemSet(42)
//	defer b.Free()
//
//	for i := range ah.List {
//		ah.List[i] = float32(i)
//	}
//
//	a.CopyFromHost(ah)
//
//	//fmt.Println("A", a.LocalCopy().List)
//	TransposeComplexYZPart(b, a)
//	bh := b.LocalCopy()
//	//fmt.Println("B", bh.List)
//
//	A := ah.Array
//	B := bh.Array
//	for c := range B {
//		for i := range B[c] {
//			for j := range B[c][i] {
//				for k := 0; k < len(B[c][i][j])/2; k++ {
//					if A[c][i][k][2*j] != B[c][i][j][2*k] ||
//						A[c][i][k][2*j+1] != B[c][i][j][2*k+1] {
//						test.Error(A[c][i][k][2*j+0], A[c][i][k][2*j+1], "!=", B[c][i][j][2*k+0], B[c][i][j][2*k+1])
//					}
//				}
//			}
//		}
//	}
//}
