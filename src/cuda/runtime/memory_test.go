// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package runtime

import (
	"runtime"
	"testing"
	"unsafe"
)

func TestMalloc(t *testing.T) {
	defer func() {
		err := recover()
		if err != nil {
			t.Error(err)
		}
	}()

	runtime.LockOSThread()
	for i := 0; i < 1024; i++ {
		pointer := Malloc(16 * 1024 * 1024)
		Free(pointer)
	}
}

func TestArray(t *testing.T) {
	defer func() {
		err := recover()
		if err != nil {
			t.Error(err)
		}
	}()

	runtime.LockOSThread()
	for i := 0; i < 1024; i++ {
		array := NewFloat32Array(16 * 1024 * 1024)
		array.Free()
	}
}

func TestMemcpy(t *testing.T) {
	defer func() {
		err := recover()
		if err != nil {
			t.Error(err)
		}
	}()

	N := (32 * 1024)
	host1 := make([]float32, N)
	for i := range host1 {
		host1[i] = 1.
	}
	host2 := make([]float32, N)
	dev1 := Malloc(4 * N)
	dev2 := Malloc(4 * N)
	Memcpy(dev1, uintptr(unsafe.Pointer(&host1[0])), 4*N, MemcpyHostToDevice)
	Memcpy(dev2, dev1, 4*N, MemcpyDeviceToDevice)
	Memcpy(uintptr(unsafe.Pointer(&host2[0])), dev2, 4*N, MemcpyDeviceToHost)
	for i := range host2 {
		if host2[i] != 1. {
			t.Fail()
		}
	}
	Free(dev1)
	Free(dev2)
}

func BenchmarkMemcpyToDevice(b *testing.B) {
	defer func() {
		err := recover()
		if err != nil {
			println(err)
		}
	}()

	b.StopTimer()
	N := (32 * 1024 * 1024)
	b.SetBytes(int64(4 * N))
	host1 := make([]float32, N)
	for i := range host1 {
		host1[i] = 1.
	}
	dev1 := Malloc(4 * N)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		Memcpy(dev1, uintptr(unsafe.Pointer(&host1[0])), 4*N, MemcpyHostToDevice)
	}
	b.StopTimer()
	Free(dev1)
}

func BenchmarkMemcpyOnDevice(b *testing.B) {
	defer func() {
		err := recover()
		if err != nil {
			println(err)
		}
	}()

	b.StopTimer()
	N := (16 * 1024 * 1024)
	b.SetBytes(int64(4 * N))
	dev1 := Malloc(4 * N)
	dev2 := Malloc(4 * N)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		Memcpy(dev2, dev1, 4*N, MemcpyDeviceToDevice)
	}
	b.StopTimer()
	Free(dev1)
	Free(dev2)
}

func TestMemcpyAsync(t *testing.T) {
	defer func() {
		err := recover()
		if err != nil {
			t.Error(err)
		}
	}()

	N := (32 * 1024)
	host1 := make([]float32, N)
	for i := range host1 {
		host1[i] = 1.
	}
	host2 := make([]float32, N)
	dev1 := Malloc(4 * N)
	dev2 := Malloc(4 * N)

	stream := StreamCreate()
	Memcpy(dev1, uintptr(unsafe.Pointer(&host1[0])), 4*N, MemcpyHostToDevice)
	MemcpyAsync(dev2, dev1, 4*N, MemcpyDeviceToDevice, stream)
	Memcpy(uintptr(unsafe.Pointer(&host2[0])), dev2, 4*N, MemcpyDeviceToHost)
	StreamSynchronize(stream)
	StreamDestroy(stream)

	for i := range host2 {
		if host2[i] != 1. {
			t.Fail()
		}
	}
	Free(dev1)
	Free(dev2)
}
