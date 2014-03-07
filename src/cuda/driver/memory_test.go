// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package driver

import (
	"fmt"
	"math"
	"testing"
	"unsafe"
)

func TestMalloc(t *testing.T) {
	for i := 0; i < 1024; i++ {
		pointer := MemAlloc(16 * 1024 * 1024)
		pointer.Free()
	}
	for i := 0; i < 1024; i++ {
		pointer := MemAlloc(16 * 1024 * 1024)
		MemFree(&pointer)
	}
	for i := 0; i < 1024; i++ {
		pointer := MemAlloc(16 * 1024 * 1024)
		MemFree(&pointer)
		pointer.Free()
	}
}

func TestMemAddressRange(t *testing.T) {
	N := 12345
	ptr := MemAlloc(int64(N))
	size, base := MemGetAddressRange(ptr)
	if size != int64(N) {
		t.Fail()
	}
	if base != ptr {
		t.Fail()
	}
	size, base = 0, DevicePtr(0)
	size, base = ptr.GetAddressRange()
	if ptr.Bytes() != int64(N) {
		t.Fail()
	}
}

func TestMemGetInfo(t *testing.T) {
	free, total := MemGetInfo()
	fmt.Println("MemGetInfo: ", free, "/", total)
	if free > total {
		t.Fail()
	}
	if total == 0 {
		t.Fail()
	}
}

func TestMemsetAsync(t *testing.T) {
	N := int64(32 * 1024)
	host1 := make([]float32, N)
	for i := range host1 {
		host1[i] = float32(i)
	}
	host2 := make([]float32, N)
	dev1 := MemAlloc(int64(4 * N))
	MemcpyHtoD(dev1, HostPtr(unsafe.Pointer(&host1[0])), 4*N)
	str := StreamCreate()
	MemsetD32Async(dev1, math.Float32bits(42), N, str)
	MemsetD32Async(dev1, math.Float32bits(21), N/2, str)
	MemcpyDtoH(HostPtr(unsafe.Pointer(&host2[0])), dev1, 4*N)
	str.Synchronize()
	(&str).Destroy()
	for i := 0; i < len(host2)/2; i++ {
		if host2[i] != 21 {
			t.Fail()
		}
	}
	for i := len(host2) / 2; i < len(host2); i++ {
		if host2[i] != 42 {
			t.Fail()
		}
	}
	dev1.Free()
}

func TestMemset(t *testing.T) {
	N := int64(32 * 1024)
	host1 := make([]float32, N)
	for i := range host1 {
		host1[i] = float32(i)
	}
	host2 := make([]float32, N)
	dev1 := MemAlloc(int64(4 * N))
	MemcpyHtoD(dev1, HostPtr(unsafe.Pointer(&host1[0])), 4*N)
	MemsetD32(dev1, math.Float32bits(42), N)
	MemsetD32(dev1, math.Float32bits(21), N/2)
	MemcpyDtoH(HostPtr(unsafe.Pointer(&host2[0])), dev1, 4*N)
	for i := 0; i < len(host2)/2; i++ {
		if host2[i] != 21 {
			t.Fail()
		}
	}
	for i := len(host2) / 2; i < len(host2); i++ {
		if host2[i] != 42 {
			t.Fail()
		}
	}
	dev1.Free()
}

func TestMemcpy(t *testing.T) {
	N := int64(32 * 1024)
	host1 := make([]float32, N)
	for i := range host1 {
		host1[i] = float32(i)
	}
	host2 := make([]float32, N)
	dev1 := MemAlloc(int64(4 * N))
	dev2 := MemAlloc(int64(4 * N))
	MemcpyHtoD(dev1, HostPtr(unsafe.Pointer(&host1[0])), 4*N)
	MemcpyDtoD(dev2, dev1, 4*N)
	MemcpyDtoH(HostPtr(unsafe.Pointer(&host2[0])), dev2, 4*N)
	for i := range host2 {
		if host2[i] != float32(i) {
			t.Fail()
		}
	}
	dev1.Free()
	dev2.Free()
}

func TestMemcpyAsync(t *testing.T) {
	N := int64(32 * 1024)
	host1 := make([]float32, N)
	for i := range host1 {
		host1[i] = float32(i)
	}
	host2 := make([]float32, N)
	dev1 := MemAlloc(int64(4 * N))
	dev2 := MemAlloc(int64(4 * N))
	stream := StreamCreate()
	MemcpyHtoDAsync(dev1, HostPtr(unsafe.Pointer(&host1[0])), 4*N, stream)
	MemcpyDtoDAsync(dev2, dev1, 4*N, stream)
	MemcpyDtoHAsync(HostPtr(unsafe.Pointer(&host2[0])), dev2, 4*N, stream)
	stream.Synchronize()
	for i := range host2 {
		if host2[i] != float32(i) {
			t.Fail()
		}
	}
	dev1.Free()
	dev2.Free()
}

func TestMemcpyAsyncRegistered(t *testing.T) {
	N := int64(32 * 1024)
	host1 := make([]float32, N)
	for i := range host1 {
		host1[i] = float32(i)
	}
	host2 := make([]float32, N)
	dev1 := MemAlloc(int64(4 * N))
	dev2 := MemAlloc(int64(4 * N))
	stream := StreamCreate()
	MemcpyHtoDAsync(dev1, HostPtr(unsafe.Pointer(&host1[0])), 4*N, stream)
	MemcpyDtoDAsync(dev2, dev1, 4*N, stream)
	MemcpyDtoHAsync(HostPtr(unsafe.Pointer(&host2[0])), dev2, 4*N, stream)
	stream.Synchronize()
	for i := range host2 {
		if host2[i] != float32(i) {
			t.Fail()
		}
	}
	dev1.Free()
	dev2.Free()
}

func BenchmarkMemcpy(b *testing.B) {
	b.StopTimer()
	N := int64(32 * 1024 * 1024)
	host1 := make([]float32, N)
	host2 := make([]float32, N)
	dev1 := MemAlloc(int64(4 * N))
	defer dev1.Free()
	dev2 := MemAlloc(int64(4 * N))
	defer dev2.Free()
	b.SetBytes(4 * N)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		MemcpyHtoD(dev1, HostPtr(unsafe.Pointer(&host1[0])), 4*N)
		MemcpyDtoD(dev2, dev1, 4*N)
		MemcpyDtoH(HostPtr(unsafe.Pointer(&host2[0])), dev2, 4*N)
	}
}

func BenchmarkMemcpyAsync(b *testing.B) {
	b.StopTimer()
	N := int64(32 * 1024 * 1024)
	host1 := make([]float32, N)
	host2 := make([]float32, N)
	dev1 := MemAlloc(int64(4 * N))
	defer dev1.Free()
	dev2 := MemAlloc(int64(4 * N))
	defer dev2.Free()
	stream := StreamCreate()
	b.SetBytes(4 * N)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		MemcpyHtoDAsync(dev1, HostPtr(unsafe.Pointer(&host1[0])), 4*N, stream)
		MemcpyDtoDAsync(dev2, dev1, 4*N, stream)
		MemcpyDtoHAsync(HostPtr(unsafe.Pointer(&host2[0])), dev2, 4*N, stream)
		stream.Synchronize()
	}
}

func MakeLockedFloat32Slice(length int) []float32 {
	PAGESIZE := 4096
	SIZEOFFLOAT := 4
	slice := make([]float32, length+2*PAGESIZE/SIZEOFFLOAT) // make it 2 pages too big

	// move first element to page lower boundary
	for int(uintptr(unsafe.Pointer(&slice[0])))%PAGESIZE != 0 {
		slice = slice[1:]
	}
	// trim to desired total length
	slice = slice[:length]

	// determine page upper boundary in bytes
	l := int64(len(slice)) * 4
	for l%int64(PAGESIZE) != 0 {
		l += 4
	}

	MemHostRegister(HostPtr(&slice[0]), l, MEMHOSTREGISTER_PORTABLE)
	return slice
}

//func BenchmarkMemcpyAsyncRegistered(b *testing.B) {
//	b.StopTimer()
//	N := int64(32 * 1024 * 1024)
//	host1 := MakeLockedFloat32Slice(int(N))//make([]float32, N)
//	host2 := make([]float32, N)
//	dev1 := MemAlloc(int64(4 * N))
//	dev2 := MemAlloc(int64(4 * N))
//	stream := StreamCreate()
//	b.SetBytes(4 * N)
//	b.StartTimer()
//	for i := 0; i < b.N; i++ {
//		MemcpyHtoDAsync(dev1, HostPtr(unsafe.Pointer(&host1[0])), 4*N, stream)
//		MemcpyAsync(dev2, dev1, 4*N, stream)
//		MemcpyDtoHAsync(HostPtr(unsafe.Pointer(&host2[0])), dev2, 4*N, stream)
//	}
//	stream.Synchronize()
//	dev1.Free()
//	dev2.Free()
//}

//func BenchmarkMemcpyToDevice(b *testing.B) {
//	b.StopTimer()
//	N := int64(32 * 1024 * 1024)
//	b.SetBytes(int64(4 * N))
//	host1 := make([]float32, N)
//	for i := range host1 {
//		host1[i] = 1.
//	}
//	dev1 := Malloc(4 * N)
//	b.StartTimer()
//	for i := 0; i < b.N; i++ {
//		Memcpy(dev1, uintptr(unsafe.Pointer(&host1[0])), 4*N, MemcpyDefault)
//	}
//	b.StopTimer()
//	Free(dev1)
//}
//
//func BenchmarkMemcpyOnDevice(b *testing.B) {
//	b.StopTimer()
//	N := int64(16 * 1024 * 1024)
//	b.SetBytes(int64(4 * N))
//	dev1 := Malloc(4 * N)
//	dev2 := Malloc(4 * N)
//	b.StartTimer()
//	for i := 0; i < b.N; i++ {
//		Memcpy(dev2, dev1, 4*N, MemcpyDefault)
//	}
//	b.StopTimer()
//	Free(dev1)
//	Free(dev2)
//}
//
//func TestMemcpyAsync(t *testing.T) {
//	N := int64(32 * 1024)
//	host1 := make([]float32, N)
//	for i := range host1 {
//		host1[i] = 1.
//	}
//	host2 := make([]float32, N)
//	dev1 := Malloc(4 * N)
//	dev2 := Malloc(4 * N)
//
//	stream := StreamCreate()
//	Memcpy(dev1, uintptr(unsafe.Pointer(&host1[0])), 4*N, MemcpyDefault)
//	MemcpyAsync(dev2, dev1, 4*N, MemcpyDefault, stream)
//	Memcpy(uintptr(unsafe.Pointer(&host2[0])), dev2, 4*N, MemcpyDefault)
//	StreamSynchronize(stream)
//	StreamDestroy(stream)
//
//	for i := range host2 {
//		if host2[i] != 1. {
//			t.Fail()
//		}
//	}
//	Free(dev1)
//	Free(dev2)
//}
