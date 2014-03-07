//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

// This files provides the package documentation
// Author: Arne Vansteenkiste

// Package with multi-GPU primitives like array allocation, copying, ...
//
// 3D Array indexing.
//
// Internal dimensions are labeled (I,J,K), I being the outermost dimension, K the innermost.
// A typical loop reads:
//	for i:=0; i<N0; i++{
//		for j:=0; j<N1; j++{
//			for k:=0; k<N2; k++{
//				...
//			}
//		}
//	}
//
// I may be a small dimension, but K must preferentially be large and align-able in CUDA memory.
//
// The underlying contiguous storage is indexed as:
// 	index := i*N1*N2 + j*N2 + k
//
// The "internal" (I,J,K) dimensions correspond to the "user" dimensions (Z,Y,X)!
// Z is typically the smallest dimension like the thickness.
//
// Slicing the geometry over multiple GPUs
//
// In the J-direction.
package gpu
