//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

//import (
//	. "hotspin-core/common"
//	"testing"
//	"hotspin-core/host"
//	"fmt"
//)

//func TestConv(t *testing.T) {
//
//	size := []int{1, 8, 8}
//	kernelSize := []int{size[0] * 1, size[1] * 2, size[2] * 2}
//
//	kernel := make([]*host.Array, 6)
//	kernel[XX] = host.NewArray(1, kernelSize)
//	kernel[XX].List[0] = 1
//
//	var conv ConvPlan
//	defer conv.Free()
//	conv.Init(size, kernel)
//
//	m := NewArray(3, size)
//	defer m.Free()
//	h := NewArray(3, size)
//	defer h.Free()
//
//	mh := m.LocalCopy()
//	mh.Array[0][0][4][4] = 1
//	fmt.Println("m", mh)
//	m.CopyFromHost(mh)
//
//	conv.Convolve(m, h)
//
//	fmt.Println("h", h.LocalCopy())
//}
