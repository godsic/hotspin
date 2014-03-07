//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Functions to read tensors as binary data. 
// Intended for fast inter-process communication or data caching,
// not as a user-friendly format to store simulation output (use omf for that).
// Uses the machine's endianess.
//
// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	"mumax/host"
	"io"
	"fmt"
	"unsafe"
)

func Read(in io.Reader) *host.Array {
	if magic := readInt(in); magic != T_MAGIC {
		panic(IOErr("Bad tensor header: " + fmt.Sprintf("%X", magic)))
	}
	rank := readInt(in)
	size := make([]int, rank)
	for i := range size {
		size[i] = readInt(in)
	}
	arr := host.NewArray(size[0], size[1:])
	readData(in, arr.List)
	return arr
}

// reads a 32-bit int
func readInt(in io.Reader) int {
	var bytes4 [4]byte
	bytes := bytes4[:]
	_, err := io.ReadFull(in, bytes)
	if err != nil {
		panic(IOErr(err.String()))
	}
	return *((*int)(unsafe.Pointer(&bytes4[0]))) // bytes-to-int conversion
}

// reads float array from binary data
func readData(in io.Reader, data []float32) {
	var bytes4 [4]byte
	bytes := bytes4[:]
	for i := range data {
		_, err := in.Read(bytes)
		if err != nil {
			panic(IOErr(err.String()))
		}
		data[i] = *((*float32)(unsafe.Pointer(&bytes4[0])))
	}
}

// converts the raw int data to a slice of 4 bytes
//func bytesToInt(bytes *[4]byte) int {
//	return *((*int)(unsafe.Pointer(bytes)))
//}
//
//// converts the raw float data to a slice of 4 bytes
//func bytesToFloat(bytes *[4]byte) float32 {
//	return *((*float32)(unsafe.Pointer(bytes)))
//}

// Reads data from the reader to the
// (already allocated) tensor.
//func (t *T) ReadFrom(in_ io.Reader) {
//	in := NewBlockingReader(in_) // Do not read incomplete slices
//	size := readHeader(in)
//	Assert(EqualSize(size, t.Size()))
//	readData(in, t.List())
//}

// Reads data from the named file to the
// (already allocated) tensor.
//func (t *T) ReadFromF(filename string) {
//	in := MustOpenRDONLY(filename)
//	defer in.Close()
//	buf := bufio.NewReader(in)
//	t.ReadFrom(buf)
//}
