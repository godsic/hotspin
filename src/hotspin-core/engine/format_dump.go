//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2012  Arne Vansteenkiste, Ben Van de Wiele and Mykola Dvornik.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Implements binary dump output,
// in the machine's endianess.
// Header: all 64-bit words:
//	magic: "#dump10\n"
//  label for "time" coordinate (8 byte string like "t" or "f")
//  time of the snapshot (double)
// 	label for "space" coordinate (like "r" or "k")
// 	cellsize
//	data rank: always 4
//	4 sizes for each direction, like: 3  128 256 1024
// 	Precission of data: 4 for float64, 8 for float64
// 	DATA
// 	crc64 of DATA and header.
// Author: Arne Vansteenkiste

import (
	. "hotspin-core/common"
	"hotspin-core/dump"
	"io"
)

func init() {
	RegisterOutputFormat(&FormatDump{})
}

type FormatDump struct{}

func (f *FormatDump) Name() string {
	return "dump"
}

func (f *FormatDump) Write(out io.Writer, q *Quant, options []string) {
	if len(options) > 0 {
		panic(InputErr("dump output format does not take options"))
	}

	data := q.Buffer(FIELD).Array
	list := q.Buffer(FIELD).List

	w := dump.NewWriter(out, dump.CRC_ENABLED)
	w.Components = len(data)
	w.MeshSize = [3]int{len(data[0]), len(data[0][0]), len(data[0][0][0])}
	w.TimeUnit = "s"
	w.Time = GetEngine().Quant("t").Scalar()
	w.MeshUnit = "m"
	w.DataLabel = q.Name()
	w.DataUnit = string(q.Unit())
	sz := GetEngine().CellSize()
	w.MeshStep = [3]float64{sz[X], sz[Y], sz[Z]} // We dont swap
	w.WriteHeader()
	w.WriteData(list)
	w.WriteHash()
}
