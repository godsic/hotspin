//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

import (
	"math/rand"
	"testing"
)

func TestAdd(test *testing.T) {
	// fail test on panic, do not crash
	defer func() {
		if err := recover(); err != nil {
			test.Error(err)
		}
	}()

	for _, size := range sizes() {

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

		Add(a, a, b)

		sum := a.LocalCopy()
		for i := range sum.List {
			if sum.List[i] != ah.List[i]+bh.List[i] {
				if !test.Failed() {
					test.Error(sum.List[i], "!=", ah.List[i], "+", bh.List[i])
				}
			}
		}
	}

}

func TestMadd(test *testing.T) {
	// fail test on panic, do not crash
	defer func() {
		if err := recover(); err != nil {
			test.Error(err)
		}
	}()

	for _, size := range sizes() {

		a := NewArray(3, size)
		defer a.Free()
		ah := a.LocalCopy()

		b := NewArray(3, size)
		defer b.Free()
		bh := b.LocalCopy()

		s := NewArray(3, size)
		defer a.Free()

		for i := range ah.List {
			ah.List[i] = rand.Float32()
			bh.List[i] = rand.Float32()
		}

		a.CopyFromHost(ah)
		b.CopyFromHost(bh)

		Madd(s, a, b, 3)

		sum := s.LocalCopy()
		for i := range sum.List {
			if !veryclose(sum.List[i], ah.List[i]+3*bh.List[i]) {
				if !test.Failed() {
					test.Error(sum.List[i], "!=", ah.List[i], "+3*", bh.List[i])
				}
			}
		}
	}
}

// based on math/all_test.go from Go release.r60, copyright the Go authors.
func tolerance(a, b, e float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}

	if a != 0 {
		e = e * a
		if e < 0 {
			e = -e
		}
	}
	return d < e
}
func close(a, b float32) bool     { return tolerance(a, b, 1e-5) }
func veryclose(a, b float32) bool { return tolerance(a, b, 4e-7) }
