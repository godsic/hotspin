//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

// Author: Arne Vansteenkiste

package engine

// Kind of quantity: VALUE, FIELD or MASK
type QuantKind int

const (
	VALUE QuantKind = 1 << (1 + iota) // A value is constant in space. Has a multiplier (to store the value) but a nil *array.
	FIELD                             // A field is space-dependent. Has no multiplier but allocated array.
	MASK                              // A mask is a point-wise multiplication of a field with a value. Has an array (possibly with NULL parts) and a multiplier.
)

// Human-readable string.
func (k QuantKind) String() string {
	switch k {
	case VALUE:
		return "VALUE"
	case FIELD:
		return "FIELD"
	case MASK:
		return "MASK"
	}
	return "illegal kind"
}
