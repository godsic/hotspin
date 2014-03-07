//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

// This file implements miscellaneous common functions.
// Author: Arne Vansteenkiste

import (
	"fmt"
	"os"
	"path"
)

// Check if array sizes are equal. Panics if arrays a and b are different. 
func CheckSize(a, b []int) {
	if len(a) != len(b) {
		panic(InputErr(fmt.Sprint("array size mismatch: ", a, "!=", b)))
	}
	for i, s := range a {
		if s != b[i] {
			panic(InputErr(fmt.Sprint("array size mismatch: ", a, "!=", b)))
		}
	}
}

// Check if array sizes are equal. Panics if arrays a and b are different. 
func CheckSize3(a, b, c []int) {
	if len(a) != len(b) || len(a) != len(c) {
		panic(InputErr(fmt.Sprint("array size mismatch: ", a, b, c)))
	}
	for i, s := range a {
		if s != b[i] || s != c[i] {
			panic(InputErr(fmt.Sprint("array size mismatch: ", a, b, c)))
		}
	}
}

// True if a and b are equal. Used to check for equal array sizes.
func EqualSize(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i, s := range a {
		if s != b[i] {
			return false
		}
	}
	return true
}

// Like fmt.Sprint with a maximum length.
// Used to limit the length of error messages that contain, e.g., a large array.
func ShortPrint(a interface{}) string {
	const MAX = 30
	str := fmt.Sprint(a)
	if len(str) > MAX {
		return str[:MAX] + "..."
	}
	return str
}

// Go equivalent of &array[index] (for a float array).
func ArrayOffset(array uintptr, index int) uintptr {
	return uintptr(array + uintptr(SIZEOF_FLOAT*index))
}

// Replaces the extension of filename by a new one.
func ReplaceExt(filename, newext string) string {
	extension := path.Ext(filename)
	return filename[:len(filename)-len(extension)] + newext
}

// Gets the directory where the executable is located.
func GetExecDir() string {
	dir, err := os.Readlink("/proc/self/exe")
	CheckErr(err, ERR_IO)
	return Parent(dir)
}

// Combines two Errors into one.
// If a and b are nil, the returned error is nil.
// If either is not nil, it is returned.
// If both are not nil, the first one is returned.
func ErrCat(a, b error) error {
	if a != nil {
		return a
	}
	if b != nil {
		return b
	}
	return nil
}
