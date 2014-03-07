//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package frontend

// This file implements a JSON-RPC-like remote procedure call protocol.
// Author: Arne Vansteenkiste

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	. "hotspin-core/common"
	"hotspin-core/host"
	"reflect"
	"runtime"
)

// An RPC server using simple JSON encoding.
// Note: we do not use the official JSON-RPC protocol
// but a simpler variant well-suited for our needs.
// Protocol:
// 	Call: ["methodname", [arg1, arg2, ...]]
// 	Response: [return_value1, return_value2, ...]
type jsonRPC struct {
	in    io.Reader
	out   io.Writer
	flush bufio.Writer

	*json.Decoder
	*json.Encoder
	receiver interface{}
	method   map[string]reflect.Value // list of methods that can be called
}

// Sets up the RPC to read JSON-encoded function calls from in and return
// the result via out. All public methods of the receiver are made accessible.
func (j *jsonRPC) Init(in io.Reader, out io.Writer, flush bufio.Writer, receiver interface{}) {
	j.in = in
	j.out = out
	j.flush = flush
	j.Decoder = json.NewDecoder(in)
	j.Encoder = json.NewEncoder(out)
	j.receiver = receiver
	j.method = make(map[string]reflect.Value)
	AddMethods(j.method, receiver)
}

// Reads JSON values from j.in, calls the corresponding functions and
// encodes the return values back to j.out.
func (j *jsonRPC) Run() {
	for {
		wbuf := new(bytes.Buffer)
		jsonc := json.NewEncoder(wbuf)

		v := new(interface{})
		err := j.Decode(v)
		if err == io.EOF {
			break
		}
		CheckErr(err, ERR_IO)

		if array, ok := (*v).([]interface{}); ok {
			Assert(len(array) == 2)
			ret := j.Call(array[0].(string), array[1].([]interface{}))
			convertOutput(ret)
			//j.Encode(ret)
			jsonc.Encode(ret)
			j.flush.WriteString(wbuf.String() + "<<< End of mumax message >>>")
			j.flush.Flush()
			//
			// wbuf now has JSON cPRC call
		} else {
			panic(IOErr(fmt.Sprint("json: ", *v)))
		}
	}
}

// Calls the function specified by funcName with the given arguments and returns the return values.
func (j *jsonRPC) Call(funcName string, args []interface{}) []interface{} {

	Debug("rpc.Call", funcName, ShortPrint(args))

	// Print which function was being called when an error occurred, for easy debugging.
	// Do not recover, however, continue on panicking.
	defer func() {
		err := recover()
		if err != nil {
			Err(fmt.Sprint("error calling ", funcName, ShortPrint(args)))
			panic(err)
		}
	}()

	f, ok := j.method[funcName]
	if !ok {
		panic(InputErr(fmt.Sprint("rpc: no such method:", funcName)))
	}

	// call
	// convert []interface{} to []reflect.Value

	argvals := make([]reflect.Value, len(args))
	for i := range argvals {
		argvals[i] = convertArg(args[i], f.Type().In(i))
	}
	retVals := f.Call(argvals)

	// convert []reflect.Value to []interface{}
	ret := make([]interface{}, len(retVals))
	for i := range retVals {
		ret[i] = retVals[i].Interface()
	}
	return ret
}

// Convert v to the specified type.
// JSON returns all numbers as float64's even when, e.g., ints are needed,
// hence such conversion. Also, convert to host.Array etc.
func convertArg(v interface{}, typ reflect.Type) reflect.Value {
	switch typ.Kind() {
	case reflect.Int:
		if float64(int(v.(float64))) != v.(float64) {
			// make sure it actually fits in an int
			panic(InputErrF("need 32-bit integer: ", v.(float64)))
		}
		return reflect.ValueOf(int(v.(float64)))
	case reflect.Float32:
		return reflect.ValueOf(float32(v.(float64)))
	}

	switch typ.String() {
	case "*host.Array":
		return reflect.ValueOf(jsonToHostArray(v))
	case "[]float32":
		return reflect.ValueOf(jsonToFloat32Array(v))
	case "[]float64":
		return reflect.ValueOf(jsonToFloat64Array(v))
	case "[]string":
		return reflect.ValueOf(jsonToStringArray(v))
	}
	return reflect.ValueOf(v) // do not convert
}

// Converts []interface{} array to []string.
// Also, converts a single string to a 1-element array.
func jsonToStringArray(v interface{}) []string {
	defer func() {
		err := recover()
		if err != nil {
			panic(IOErr(fmt.Sprint("Error parsing json array: ", ShortPrint(v), "\ncause: ", err)))
		}
	}()

	switch v.(type) {
	case string:
		return []string{v.(string)}
	case []interface{}:
		varray := v.([]interface{})
		array := make([]string, len(varray))
		for i := range array {
			array[i] = varray[i].(string)
		}
		return array
	}
	panic(IOErr("Expected string or string array, got: " + ShortPrint(v) + " of type: " + reflect.TypeOf(v).String()))
	return nil //silence 6g
}

// Converts []interface{} array to []float32.
// Also, converts a single float32 to a 1-element array.
func jsonToFloat32Array(v interface{}) []float32 {
	defer func() {
		err := recover()
		if err != nil {
			panic(IOErr(fmt.Sprint("Error parsing json array: ", ShortPrint(v), "\ncause: ", err)))
		}
	}()

	switch v.(type) {
	case float64:
		return []float32{float32(v.(float64))}
	case []interface{}:
		varray := v.([]interface{})
		array := make([]float32, len(varray))
		for i := range array {
			array[i] = float32(varray[i].(float64))
		}
		return array
	}
	panic(IOErr("Expected float32 or float32 array, got: " + ShortPrint(v) + " of type: " + reflect.TypeOf(v).String()))
	return nil //silence 6g
}

// Converts []interface{} array to []float64.
// Also, converts a single float64 to a 1-element array.
func jsonToFloat64Array(v interface{}) []float64 {
	defer func() {
		err := recover()
		if err != nil {
			panic(IOErr(fmt.Sprint("Error parsing json array: ", ShortPrint(v), "\ncause: ", err)))
		}
	}()

	switch v.(type) {
	case float64:
		return []float64{v.(float64)}
	case []interface{}:
		varray := v.([]interface{})
		array := make([]float64, len(varray))
		for i := range array {
			array[i] = varray[i].(float64)
		}
		return array
	}
	panic(IOErr("Expected float64 or float64 array, got: " + ShortPrint(v) + " of type: " + reflect.TypeOf(v).String()))
	return nil //silence 6g
}

// Converts a json vector array to a host.Array.
// Also swaps XYZ - ZYX convention
// TODO: works only for 4D vector arrays
func jsonToHostArray(v interface{}) *host.Array {
	defer func() {
		err := recover()
		if err != nil {
			panic(IOErr(fmt.Sprint("Error parsing json array: ", ShortPrint(v), "\ncause: ", err)))
		}
	}()

	err := false
	// determine array size as {len(v), len(v[0]), len(v[0][0]), ...}
	var size [4]int
	v2 := v
	for i := range size {
		if arr, ok := v2.([]interface{}); ok {
			size[i] = len(arr)
			if size[i] == 0 {
				err = true
				break
			}
			v2 = arr[0]
		} else {
			err = true
			break
		}
	}

	if err {
		panic(IOErr(fmt.Sprint("Array with invalid size:", ShortPrint(v))))
	}

	size3D := size[1:]
	arr := host.NewArray(size[0], []int{size3D[X], size3D[Y], size3D[Z]})
	a := arr.Array
	va := v.([]interface{})
	for c := range a {
		va_c := va[c].([]interface{})
		for i := range a[c] {
			va_ci := va_c[i].([]interface{})
			for j := range a[c][i] {
				va_cij := va_ci[j].([]interface{})
				for k := range a[c][i][j] {
					a[c][i][j][k] = float32(va_cij[k].(float64)) // convert XYZ-ZYX, works only for 3D
				}
			}
		}
	}
	runtime.GC() // a LOT of garbage has been made
	return convertXYZ(arr)
}

// convert mumax return values to types suited for json encoding
// most values remain the same, but host.Array gets converted
// to [][][][][]float32 and transposed into ZYX userspace
func convertOutput(vals []interface{}) {
	for i, v := range vals {
		switch v.(type) {
		default:
			vals[i] = v
		case *host.Array:
			vals[i] = convertXYZ(v.(*host.Array)).Array
		}
	}
}

// Convert mumax's internal ZYX convention to userspace XYZ.
func convertXYZ(arr *host.Array) *host.Array {
	s := arr.Size3D
	n := arr.NComp()
	a := arr.Array
	transp := host.NewArray(n, []int{s[Z], s[Y], s[X]})
	t := transp.Array
	for c := 0; c < n; c++ {
		for i := 0; i < s[X]; i++ {
			for j := 0; j < s[Y]; j++ {
				for k := 0; k < s[Z]; k++ {
					t[(n-1)-c][k][j][i] = a[c][i][j][k]
				}
			}
		}
	}
	runtime.GC() // a LOT of garbage has been made
	return transp
}
