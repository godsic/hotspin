//  This file is part of MuMax, a high-performance micromagnetic simulator
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

import (
	"fmt"
	. "hotspin-core/common"
	"hotspin-core/host"
	"math"
)

// A Reductor stores the necessary buffers to reduce data on the multi-GPU.
// It can be used to sum data, take minima, maxima, etc...
type Reductor struct {
	devbuffer          Array
	hostbuffer         host.Array
	blocks, threads, N int
	size4D             [4]int
}

// Initiate buffers to reduce an array of given size
func (r *Reductor) Init(nComp int, size []int) {
	N := (nComp * Prod(size)) // N floats per device
	Assert(N > 1)
	Assert(len(size) == 3)

	r.threads = maxThreadsPerBlock
	Debug("threads:", r.threads)
	if r.threads > 1024 {
		r.threads = 1024
	} // TODO reduce.cu only templated up to 1024

	for N <= r.threads {
		r.threads /= 2
	}
	r.blocks = DivUp(N, r.threads*2)
	r.N = N

	bufsize := []int{1, 1, r.blocks}
	(&(r.devbuffer)).Free()            // Re-initialization should not leak memory
	r.devbuffer.Init(1, bufsize, true) // true=do alloc
	r.hostbuffer.Init(1, bufsize)
	r.size4D[0] = nComp
	r.size4D[1] = size[0]
	r.size4D[2] = size[1]
	r.size4D[3] = size[2]
}

// Make reductor to reduce an array of given size
func NewReductor(nComp int, size []int) *Reductor {
	r := new(Reductor)
	r.Init(nComp, size)
	return r
}

// Frees the GPU buffer storage.
func (r *Reductor) Free() {
	(&(r.devbuffer)).Free()
	r.blocks = 0
	r.threads = 0
	r.N = 0
}

// Takes the sum of all elements of the array.
func (r *Reductor) Sum(in *Array) float32 {
	r.checkSize(in)
	PartialSum(in, (&r.devbuffer), r.blocks, r.threads, r.N)
	// reduce further on CPU
	(&r.devbuffer).CopyToHost(&r.hostbuffer)
	var sum float32
	for _, num := range r.hostbuffer.List {
		sum += num
	}
	return sum
}

// Takes the dot product of all elements of the arrays.
func (r *Reductor) Dot(in1, in2 *Array) float32 {
	r.checkSize(in1)
	r.checkSize(in2)
	PartialSDot(in1, in2, (&r.devbuffer), r.blocks, r.threads, r.N)
	// reduce further on CPU
	(&r.devbuffer).CopyToHost(&r.hostbuffer)
	var sum float32
	for _, num := range r.hostbuffer.List {
		sum += num
	}
	return sum
}

// Takes the maximum of all elements of the array.
func (r *Reductor) Max(in *Array) float32 {
	r.checkSize(in)
	PartialMax(in, (&r.devbuffer), r.blocks, r.threads, r.N)
	// reduce further on CPU
	(&r.devbuffer).CopyToHost(&r.hostbuffer)
	max := r.hostbuffer.List[0]
	for _, num := range r.hostbuffer.List {
		if num > max {
			max = num
		}
	}
	return max
}

// Takes the minimum of all elements of the array.
func (r *Reductor) Min(in *Array) float32 {
	r.checkSize(in)
	PartialMin(in, (&r.devbuffer), r.blocks, r.threads, r.N)
	// reduce further on CPU
	(&r.devbuffer).CopyToHost(&r.hostbuffer)
	min := r.hostbuffer.List[0]
	for _, num := range r.hostbuffer.List {
		if num < min {
			min = num
		}
	}
	return min
}

// Takes the maximum of absolute values of all elements of the array.
func (r *Reductor) MaxAbs(in *Array) float32 {
	r.checkSize(in)
	PartialMaxAbs(in, (&r.devbuffer), r.blocks, r.threads, r.N)
	// reduce further on CPU
	(&r.devbuffer).CopyToHost(&r.hostbuffer)
	max := r.hostbuffer.List[0] // all values are already positive
	for _, num := range r.hostbuffer.List {
		if num > max {
			max = num
		}
	}
	return max
}

// Takes the maximum absolute difference between the elements of a and b.
func (r *Reductor) MaxDiff(a, b *Array) float32 {
	r.checkSize(a)
	r.checkSize(b)
	PartialMaxDiff(a, b, (&r.devbuffer), r.blocks, r.threads, r.N)
	// reduce further on CPU
	(&r.devbuffer).CopyToHost(&r.hostbuffer)
	max := r.hostbuffer.List[0] // all values are already positive
	for _, num := range r.hostbuffer.List {
		if num > max {
			max = num
		}
	}
	return max
}

// Takes the maximum absolute sum between the elements of a and b.
func (r *Reductor) MaxSum(a, b *Array) float32 {
	r.checkSize(a)
	r.checkSize(b)
	PartialMaxSum(a, b, (&r.devbuffer), r.blocks, r.threads, r.N)
	// reduce further on CPU
	(&r.devbuffer).CopyToHost(&r.hostbuffer)
	max := r.hostbuffer.List[0] // all values are already positive
	for _, num := range r.hostbuffer.List {
		if num > max {
			max = num
		}
	}
	return max
}

// Takes the maximum norm of a 3-component (vector) array.
func (r *Reductor) MaxNorm(a *Array) float32 {
	r.checkSize(&a.Comp[0])
	Assert(a.NComp() == 3)
	PartialMaxNorm3Sq(&a.Comp[X], &a.Comp[Y], &a.Comp[Z], (&r.devbuffer), r.blocks, r.threads, r.N)
	// reduce further on CPU
	(&r.devbuffer).CopyToHost(&r.hostbuffer)
	max := r.hostbuffer.List[0] // all values are already positive
	for _, num := range r.hostbuffer.List {
		if num > max {
			max = num
		}
	}
	return float32(math.Sqrt(float64(max)))
}

// Takes the maximum norm of the difference between two 3-component (vector) arrays.
func (r *Reductor) MaxNormDiff(a, b *Array) float32 {
	r.checkSize(a)
	Assert(a.NComp() == 3)
	PartialMaxNorm3SqDiff(&a.Comp[X], &a.Comp[Y], &a.Comp[Z], &b.Comp[X], &b.Comp[Y], &b.Comp[Z], (&r.devbuffer), r.blocks, r.threads, r.N)
	// reduce further on CPU
	(&r.devbuffer).CopyToHost(&r.hostbuffer)
	max := r.hostbuffer.List[0] // all values are already positive
	for _, num := range r.hostbuffer.List {
		if num > max {
			max = num
		}
	}
	return float32(math.Sqrt(float64(max)))
}

// INTERNAL: Make sure in has the right size for this reductor
func (r *Reductor) checkSize(in *Array) {
	for i, s := range r.size4D {
		if s != in.size4D[i] {
			panic(Bug(fmt.Sprint("Reductor size mismatch", in.size4D, "!=", r.size4D)))
		}
	}

}
