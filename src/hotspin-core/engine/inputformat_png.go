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
	"image/png"
	. "hotspin-core/common"
	"hotspin-core/host"
	"os"
)

func init() {
	RegisterInputFormat(".png", ReadPNG)
}

func ReadPNG(fname string) *host.Array {
	in, err := os.Open(fname)
	CheckIO(err)

	img, err2 := png.Decode(in)
	CheckIO(err2)

	width := img.Bounds().Max.X
	height := img.Bounds().Max.Y

	inside := host.NewArray(1, []int{1, height, width})

	for i := 0; i < height; i++ {
		for j := 0; j < width; j++ {
			r, g, b, _ := img.At(j, height-1-i).RGBA()
			if r+g+b < (0xFFFF*3)/2 {
				inside.Array[0][0][i][j] = 1
			}
		}
	}
	return inside
}
