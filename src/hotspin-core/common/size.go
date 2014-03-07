//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

// Author: Arne Vansteenkiste

import (
	"fmt"
)

// fmt.Stringer for slices representing a size.
type Size []int

// Human readable size "Z x Y x X"
func (s Size) String() string {
	str := fmt.Sprint(s[0])
	for _, size := range s[1:] {
		str = fmt.Sprint(size, "x", str)
	}
	return str
}
