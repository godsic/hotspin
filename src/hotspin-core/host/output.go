//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package host

// Auhtor: Arne Vansteenkiste

import (
	"fmt"
	. "hotspin-core/common"
	"io"
	"unsafe"
)

func (tens *Array) WriteAscii(out io.Writer) {
	data := tens.Array
	gridsize := tens.Size3D

	// Here we loop over X,Y,Z, not Z,Y,X, because
	// internal in C-order == external in Fortran-order
	for i := 0; i < gridsize[X]; i++ {
		for j := 0; j < gridsize[Y]; j++ {
			for k := 0; k < gridsize[Z]; k++ {
				for c := 0; c < tens.NComp(); c++ {
					_, err := fmt.Fprint(out, data[SwapIndex(c, tens.NComp())][i][j][k], " ") // converts to user space.
					if err != nil {
						panic(IOErr(err.Error()))
					}
				}
				//_, err := fmt.Fprint(out, "\n")
				//if err != nil {
				//	panic(IOErr(err.String()))
				//}
			}
			_, err := fmt.Fprint(out, "\n")
			if err != nil {
				panic(IOErr(err.Error()))
			}
		}
		_, err := fmt.Fprint(out, "\n")
		if err != nil {
			panic(IOErr(err.Error()))
		}
	}
}

func (t *Array) WriteBinary(out io.Writer) {
	out.Write(intToBytes(T_MAGIC))
	out.Write(intToBytes(4)) // Rank is always 4
	for _, s := range t.Size4D {
		out.Write(intToBytes(s))
	}
	for _, f := range t.List {
		out.Write((*[4]byte)(unsafe.Pointer(&f))[:]) // FloatToBytes() inlined for performance.
	}
}

const (
	T_MAGIC = 0x0A317423 // First 32-bit word of tensor blob. Identifies the format. Little-endian ASCII for "#t1\n"
)

// Converts the raw int data to a slice of 4 bytes
func intToBytes(i int) []byte {
	return (*[4]byte)(unsafe.Pointer(&i))[:]
}

// Converts the raw float data to a slice of 4 bytes
func floatToBytes(f float64) []byte {
	return (*[4]byte)(unsafe.Pointer(&f))[:]
}
