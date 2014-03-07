//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2012  Arne Vansteenkiste, Ben Van de Wiele and Mykola Dvornik.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Auhtor: Arne Vansteenkiste

import ()

import (
	. "hotspin-core/common"
	"hotspin-core/dump"
	"hotspin-core/host"
	"os"
)

func init() {
	RegisterInputFormat(".dump", ReadDump)
}

func ReadDump(fname string) *host.Array {
	in, err := os.Open(fname)
	CheckIO(err)
	r := dump.NewReader(in, true)
	err = r.Read()
	CheckIO(err)
	meshsize := []int{r.Frame.MeshSize[0], r.Frame.MeshSize[1], r.Frame.MeshSize[2]}
	return host.NewArrayFromList(r.Frame.Components, meshsize, r.Frame.Data)
}
