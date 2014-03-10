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
	"math/rand"
	. "hotspin-core/common"
	"hotspin-core/host"
	"testing"
)

func TestReduceSum(test *testing.T) {
	// fail test on panic, do not crash
	defer func() {
		if err := recover(); err != nil {
			test.Error(err)
		}
	}()

	size := []int{8, 16, 32}
	a := NewArray(3, size)
	defer a.Free()
	ah := a.LocalCopy()

	for i := range ah.List {
		ah.List[i] = rand.Float32()
	}
	a.CopyFromHost(ah)

	var cpusum float64
	for _, num := range ah.List {
		cpusum += float64(num)
	}

	red := NewReductor(a.NComp(), a.Size3D())
	defer red.Free()
	gpusum := red.Sum(a)

	if !close(gpusum, float64(cpusum)) {
		test.Error("Reduce sum cpu=", cpusum, "gpu=", gpusum)
	}
}

func TestReduceMax(test *testing.T) {
	// fail test on panic, do not crash
	defer func() {
		if err := recover(); err != nil {
			test.Error(err)
		}
	}()

	size := []int{8, 16, 32}
	a := NewArray(3, size)
	defer a.Free()
	ah := a.LocalCopy()

	for i := range ah.List {
		ah.List[i] = rand.Float32()
	}
	a.CopyFromHost(ah)

	cpumax := ah.List[0]
	for _, num := range ah.List {
		if num > cpumax {
			cpumax = num
		}
	}

	red := NewReductor(a.NComp(), a.Size3D())
	defer red.Free()
	gpumax := red.Max(a)

	if gpumax != cpumax {
		test.Error("Reduce max cpu=", cpumax, "gpu=", gpumax)
	}
}

func TestReduceMin(test *testing.T) {
	// fail test on panic, do not crash
	defer func() {
		if err := recover(); err != nil {
			test.Error(err)
		}
	}()

	size := []int{8, 16, 32}
	a := NewArray(3, size)
	defer a.Free()
	ah := a.LocalCopy()

	for i := range ah.List {
		ah.List[i] = rand.Float32()
	}
	a.CopyFromHost(ah)

	cpumin := ah.List[0]
	for _, num := range ah.List {
		if num < cpumin {
			cpumin = num
		}
	}

	red := NewReductor(a.NComp(), a.Size3D())
	defer red.Free()
	gpumin := red.Min(a)

	if gpumin != cpumin {
		test.Error("Reduce min cpu=", cpumin, "gpu=", gpumin)
	}
}

func TestReduceMaxAbs(test *testing.T) {
	// fail test on panic, do not crash
	defer func() {
		if err := recover(); err != nil {
			test.Error(err)
		}
	}()

	size := []int{8, 16, 32}
	a := NewArray(3, size)
	defer a.Free()
	ah := a.LocalCopy()

	for i := range ah.List {
		ah.List[i] = -rand.Float32()
	}
	a.CopyFromHost(ah)

	cpumax := ah.List[0]
	for _, num := range ah.List {
		if Abs32(num) > cpumax {
			cpumax = Abs32(num)
		}
	}

	red := NewReductor(a.NComp(), a.Size3D())
	defer red.Free()
	gpumax := red.MaxAbs(a)

	if gpumax != cpumax {
		test.Error("Reduce maxabs cpu=", cpumax, "gpu=", gpumax)
	}
}

func TestReduceMaxDiff(test *testing.T) {
	// fail test on panic, do not crash
	defer func() {
		if err := recover(); err != nil {
			test.Error(err)
		}
	}()

	size := []int{8, 16, 32}
	a := NewArray(3, size)
	defer a.Free()
	ah := a.LocalCopy()
	b := NewArray(3, size)
	defer b.Free()
	bh := b.LocalCopy()

	for i := range ah.List {
		ah.List[i] = rand.Float32()
		bh.List[i] = rand.Float32()
	}
	a.CopyFromHost(ah)
	b.CopyFromHost(bh)

	cpumax := float64(0)
	for i, _ := range ah.List {
		if Abs32(ah.List[i]-bh.List[i]) > cpumax {
			cpumax = Abs32(ah.List[i] - bh.List[i])
		}
	}

	red := NewReductor(a.NComp(), a.Size3D())
	defer red.Free()
	gpumax := red.MaxDiff(a, b)

	if gpumax != cpumax {
		test.Error("Reduce maxabs cpu=", cpumax, "gpu=", gpumax)
	}
}

func BenchmarkReduceSum(b *testing.B) {
	b.StopTimer()

	size := bigsize()
	a := NewArray(3, size)
	b.SetBytes(int64(a.Len()) * SIZEOF_FLOAT)
	defer a.Free()
	red := NewReductor(a.NComp(), a.Size3D())
	defer red.Free()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		red.Sum(a)
	}
}

func BenchmarkReduceSumCPU(b *testing.B) {

	b.StopTimer()

	size := bigsize()
	a := host.NewArray(3, size)
	b.SetBytes(int64(a.Len()) * SIZEOF_FLOAT)
	b.StartTimer()
	for i := 0; i < b.N; i++ {

		var cpusum float64
		for _, num := range a.List {
			cpusum += float64(num)
		}

	}
}

func BenchmarkReduceMaxDiff(b *testing.B) {
	b.StopTimer()

	size := bigsize()
	a := NewArray(3, size)
	a2 := NewArray(3, size)
	b.SetBytes(2 * int64(a.Len()) * SIZEOF_FLOAT)
	defer a.Free()
	defer a2.Free()
	red := NewReductor(a.NComp(), a.Size3D())
	defer red.Free()
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		red.MaxDiff(a2, a)
	}
}
