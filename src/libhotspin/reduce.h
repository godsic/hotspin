//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.


/*
 * @file reduce.h
 *
 * @author Arne Vansteenkiste
 * @author Ben Van de Wiele
 */
#ifndef REDUCE_H
#define REDUCE_H

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif


/// Multi-GPU partial sum.
/// @param input input data parts for each GPU. each array size is NPerGPU
/// @param output partially summed data for each GPU, usually copied and reduced further on the CPU. size of each array = blocksPerGPU
/// @param blocksPerGPU number of thread blocks per GPU. blocksPerGPU = divUp(NPerGPU, threadsPerBlockPerGPU*2)
/// @param threadsPerBlockPerGPU use this many threads per GPU thread block. @warning must be < NPerGPU
/// @param NPerGPU size of input data per GPU, must be > threadsPerBlockPerGPU
/// @param streams array of cuda streams on each device for async execution
DLLEXPORT void partialSumAsync(double* input, double* output, int blocksPerGPU, int threadsPerBlockPerGPU, int NPerGPU, CUstream streams);


/// Multi-GPU Dot product
/// @param input1, input2: input data parts for each GPU. each array size is NPerGPU
/// @param output partially dot-producted data for each GPU, usually copied and reduced further on the CPU. size of each array = blocksPerGPU
/// @param blocksPerGPU number of thread blocks per GPU. blocksPerGPU = divUp(NPerGPU, threadsPerBlockPerGPU*2)
/// @param threadsPerBlockPerGPU use this many threads per GPU thread block. @warning must be < NPerGPU
/// @param NPerGPU size of input data per GPU, must be > threadsPerBlockPerGPU
/// @param streams array of cuda streams on each device for async execution
DLLEXPORT void partialSDotAsync(double* input1, double* input2, double* output, int blocks, int threadsPerBlock, int N, CUstream stream);

/// Multi-GPU partial maximum.
/// @param input input data parts for each GPU. each array size is NPerGPU
/// @param output partially summed data for each GPU, usually copied and reduced further on the CPU. size of each array = blocksPerGPU
/// @param blocksPerGPU number of thread blocks per GPU. blocksPerGPU = divUp(NPerGPU, threadsPerBlockPerGPU*2)
/// @param threadsPerBlockPerGPU use this many threads per GPU thread block. @warning must be < NPerGPU
/// @param NPerGPU size of input data per GPU, must be > threadsPerBlockPerGPU
/// @param streams array of cuda streams on each device for async execution
DLLEXPORT void partialMaxAsync(double* input, double* output, int blocksPerGPU, int threadsPerBlockPerGPU, int NPerGPU, CUstream streams);


/// Multi-GPU partial minimum.
/// @param input input data parts for each GPU. each array size is NPerGPU
/// @param output partially summed data for each GPU, usually copied and reduced further on the CPU. size of each array = blocksPerGPU
/// @param blocksPerGPU number of thread blocks per GPU. blocksPerGPU = divUp(NPerGPU, threadsPerBlockPerGPU*2)
/// @param threadsPerBlockPerGPU use this many threads per GPU thread block. @warning must be < NPerGPU
/// @param NPerGPU size of input data per GPU, must be > threadsPerBlockPerGPU
/// @param streams array of cuda streams on each device for async execution
DLLEXPORT void partialMinAsync(double* input, double* output, int blocksPerGPU, int threadsPerBlockPerGPU, int NPerGPU, CUstream streams);


/// Multi-GPU partial maximum of absolute values.
/// @param input input data parts for each GPU. each array size is NPerGPU
/// @param output partially summed data for each GPU, usually copied and reduced further on the CPU. size of each array = blocksPerGPU
/// @param blocksPerGPU number of thread blocks per GPU. blocksPerGPU = divUp(NPerGPU, threadsPerBlockPerGPU*2)
/// @param threadsPerBlockPerGPU use this many threads per GPU thread block. @warning must be < NPerGPU
/// @param NPerGPU size of input data per GPU, must be > threadsPerBlockPerGPU
/// @param streams array of cuda streams on each device for async execution
DLLEXPORT void partialMaxAbsAsync(double* input, double* output, int blocksPerGPU, int threadsPerBlockPerGPU, int NPerGPU, CUstream streams);


/// Multi-GPU partial maximum difference between arrays (max(abs(a[i]-b[i]))).
/// @param input input data parts for each GPU. each array size is NPerGPU
/// @param output partially summed data for each GPU, usually copied and reduced further on the CPU. size of each array = blocksPerGPU
/// @param blocksPerGPU number of thread blocks per GPU. blocksPerGPU = divUp(NPerGPU, threadsPerBlockPerGPU*2)
/// @param threadsPerBlockPerGPU use this many threads per GPU thread block. @warning must be < NPerGPU
/// @param NPerGPU size of input data per GPU, must be > threadsPerBlockPerGPU
/// @param streams array of cuda streams on each device for async execution
DLLEXPORT void partialMaxDiffAsync(double* a, double* b, double* output, int blocksPerGPU, int threadsPerBlockPerGPU, int NPerGPU, CUstream streams);

/// Multi-GPU partial maximum sum between arrays (max(abs(a[i]+b[i]))).
/// @param input input data parts for each GPU. each array size is NPerGPU
/// @param output partially summed data for each GPU, usually copied and reduced further on the CPU. size of each array = blocksPerGPU
/// @param blocksPerGPU number of thread blocks per GPU. blocksPerGPU = divUp(NPerGPU, threadsPerBlockPerGPU*2)
/// @param threadsPerBlockPerGPU use this many threads per GPU thread block. @warning must be < NPerGPU
/// @param NPerGPU size of input data per GPU, must be > threadsPerBlockPerGPU
/// @param streams array of cuda streams on each device for async execution
DLLEXPORT void partialMaxSumAsync(double* a, double* b, double* output, int blocksPerGPU, int threadsPerBlockPerGPU, int NPerGPU, CUstream streams);


/// Multi-GPU partial maximum Euclidian norm squared of 3-vector(max(x[i]*2+y[i]*2+z[i]*2)).
/// @param x input vector x-component parts for each GPU. each array size is NPerGPU
/// @param y input vector y-component parts for each GPU. each array size is NPerGPU
/// @param z input vector z-component parts for each GPU. each array size is NPerGPU
/// @param output partially summed data for each GPU, usually copied and reduced further on the CPU. size of each array = blocksPerGPU
/// @param blocksPerGPU number of thread blocks per GPU. blocksPerGPU = divUp(NPerGPU, threadsPerBlockPerGPU*2)
/// @param threadsPerBlockPerGPU use this many threads per GPU thread block. @warning must be < NPerGPU
/// @param NPerGPU size of input data per GPU, must be > threadsPerBlockPerGPU
/// @param streams array of cuda streams on each device for async execution
DLLEXPORT void partialMaxNorm3SqAsync(double* x, double* y, double* z, double* output, int blocksPerGPU, int threadsPerBlockPerGPU, int NPerGPU, CUstream streams);


/// Multi-GPU partial maximum Euclidian norm squared of 3-vector(max( (x1[i]-x2[i])*2 + (y1[i]-y2[i])*2 + (z1[i]-z2[i])*2) ).
/// @param x1 input vector 1 x-component parts for each GPU. each array size is NPerGPU
/// @param y1 input vector 1 y-component parts for each GPU. each array size is NPerGPU
/// @param z1 input vector 1 z-component parts for each GPU. each array size is NPerGPU
/// @param x2 input vector 2 x-component parts for each GPU. each array size is NPerGPU
/// @param y2 input vector 2 y-component parts for each GPU. each array size is NPerGPU
/// @param z2 input vector 2 z-component parts for each GPU. each array size is NPerGPU
/// @param output partially summed data for each GPU, usually copied and reduced further on the CPU. size of each array = blocksPerGPU
/// @param blocksPerGPU number of thread blocks per GPU. blocksPerGPU = divUp(NPerGPU, threadsPerBlockPerGPU*2)
/// @param threadsPerBlockPerGPU use this many threads per GPU thread block. @warning must be < NPerGPU
/// @param NPerGPU size of input data per GPU, must be > threadsPerBlockPerGPU
/// @param streams array of cuda streams on each device for async execution
DLLEXPORT void partialMaxNorm3SqDiffAsync(double* x1, double* y1, double* z1, double* x2, double* y2, double* z2, double* output, int blocksPerGPU, int threadsPerBlockPerGPU, int NPerGPU, CUstream streams);


#ifdef __cplusplus
}
#endif
#endif
