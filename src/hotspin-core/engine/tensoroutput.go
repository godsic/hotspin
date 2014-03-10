//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Functions to write tensors as binary data. 
// Intended for fast inter-process communication or data caching,
// not as a user-friendly format to store simulation output (use omf for that).
// Uses the machine's endianess.
//
// Author: Arne Vansteenkiste

import (
	"io"
	. "hotspin-core/common"
	"hotspin-core/host"
	"unsafe"
)

const (
	T_MAGIC = 0x0A317423 // First 32-bit word of tensor blob. Identifies the format. Little-endian ASCII for "#t1\n"
)

// Writes the array
func Write(out io.Writer, a *host.Array) {
	writeInt(out, T_MAGIC)
	writeInt(out, a.Rank())
	for _, s := range a.Size {
		writeInt(out, s)
	}
	writeData(out, a.List)
}

// writes an integer
func writeInt(out io.Writer, i int) {
	_, err := out.Write((*[4]byte)(unsafe.Pointer(&i))[:])
	if err != nil {
		panic(IOErr(err.Error()))
	}
}

// block this many float's for binary I/O
const block = 256

// writes the array to the writer, in binary format
func writeData(out io.Writer, data []float64) {
	// first write as much data as possible data in large blocks
	count := 0
	for i := 0; i < len(data); i += block {
		_, err := out.Write((*[4 * block]byte)(unsafe.Pointer(&data[i]))[:])
		if err != nil {
			panic(IOErr(err.Error()))
		}
		count += block
	}
	// then write the remainder as a few individual floats
	for i := count; i < len(data); i++ {
		_, err := out.Write((*[4]byte)(unsafe.Pointer(&data[i]))[:])
		if err != nil {
			panic(IOErr(err.Error()))
		}
	}
}

func writeFloat(out io.Writer, f float64) {
	_, err := out.Write((*[4]byte)(unsafe.Pointer(&f))[:])
	if err != nil {
		panic(IOErr(err.Error()))
	}
}

func writeDouble(out io.Writer, f float64) {
	_, err := out.Write((*[8]byte)(unsafe.Pointer(&f))[:])
	if err != nil {
		panic(IOErr(err.Error()))
	}
}

// Converts the raw int data to a slice of 4 bytes
//func IntToBytes(i int) []byte {
//	return (*[4]byte)(unsafe.Pointer(&i))[:]
//}
//
//// Converts the raw float data to a slice of 4 bytes
//func FloatToBytes(f float64) []byte {
//	return (*[4]byte)(unsafe.Pointer(&f))[:]
//}

// TODO: 
// also necessary to implement io.WriterTo, ReaderFrom
//func (t *T) WriteTo(out io.Writer) {
//	Write(out, t)
//}
// Utility function, reads from a named file instead of io.Reader.
//func WriteF(filename string, t host.Array) {
//	out := MustOpenWRONLY(filename)
//	defer out.Close()
//	bufout := bufio.NewWriter(out)
//	defer bufout.Flush()
//	Write(bufout, t)
//}
