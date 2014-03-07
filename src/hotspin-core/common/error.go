//  This file is part of MuMax, a high-perfomrance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

// This file implements mumax error types.
// InputErr: illegal input, e.g. in the input file.
// IOErr: I/O error, e.g. file not found
// Bug: this is not the user's fault. A crash report should be generated.
// Author: Arne Vansteenkiste

import "fmt"

// We define different error types so a recover() after
// panic() can determine (with a type assertion)
// what kind of error happened. 
// Only a Bug error causes a bug report and stack dump,
// other errors are the user's fault and do not trigger
// a stack dump.

// The input file contains illegal input
type InputErr string

func (e InputErr) String() string {
	return string(e)
}

// Like fmt.Sprint() but inserts spaces between the arguments.
func sprint(msg ...interface{}) string {
	str := fmt.Sprintln(msg...) // inserts spaces
	return str[:len(str)-1]     // omit newline
}

// Shorthand for InputErr(fmt.Sprint(msg...))
func InputErrF(msg ...interface{}) InputErr {
	return InputErr(sprint(msg...))
}

// Empty function implements interface{InputErr()}
func (e *InputErr) InputErr() {
}

// A file could not be read/written
type IOErr string

func (e IOErr) String() string {
	return string(e)
}

// Shorthand for IOErr(fmt.Sprint(msg...))
func IOErrF(msg ...interface{}) IOErr {
	return IOErr(sprint(msg...))
}

// Empty function implements interface{IOErr()}
func (e *IOErr) IOErr() {
}

// An unexpected error occurred which should be reported
type Bug string

func (e Bug) String() string {
	return string(e)
}

// Shorthand for Bug(fmt.Sprint(msg...))
func BugF(msg ...interface{}) Bug {
	return Bug(sprint(msg...))
}

// Empty function implements interface{Bug()}
func (e *Bug) Bug() {
}

// Exits with the exit code if the error is not nil.
func CheckErr(err error, code int) {
	if err == nil {
		return
	}
	switch code {
	default:
		panic(err)
	case ERR_IO:
		panic(IOErr(err.Error()))
	case ERR_INPUT:
		panic(InputErr(err.Error()))
	case ERR_BUG:
		panic(Bug(err.Error()))
	}
}

// Raises IOErr if err != nil
func CheckIO(err error) {
	CheckErr(err, ERR_IO)
}

// Raises InputErr if err != nil
func CheckInput(err error) {
	CheckErr(err, ERR_INPUT)
}

// Raises Bug if err != nil
func CheckBug(err error) {
	CheckErr(err, ERR_BUG)
}

//func Exit(status int) {
//	Log("Exiting with status", status, ":", ErrString[status])
//	os.Exit(status)
//}

// Exit error code
const (
	ERR_NONE  = iota // Successful exit, no error.
	ERR_IO           // IO error
	ERR_INPUT        // Illegal input
	ERR_CUDA         // CUDA error
	ERR_BUG          // Bug
	ERR_PANIC        // Unspecified panic
)

// Human readable description of exit codes
var ErrString []string = []string{"Success", "I/O error", "Illegal input", "CUDA error", "Bug", "Unexpected panic"}
