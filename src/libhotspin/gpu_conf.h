/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

/**
 * @file
 * Create and check CUDA thread launch configurations
 *
 * @todo Needs to be completely re-worked, with macros to define the indices etc.
 *
 * @author Arne Vansteenkiste
 */
#ifndef gpu_conf_h
#define gpu_conf_h

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BLOCKSIZE3DX 2
#define BLOCKSIZE3DY 2
#define BLOCKSIZE3DZ 64

#define BLOCKSIZE2DX 1
#define BLOCKSIZE2DY 8
#define BLOCKSIZE2DZ 64

#define BLOCKSIZE1DX 1
#define BLOCKSIZE1DY 16
#define BLOCKSIZE1DZ 64

/// Returns the maximum number of threads per block for this GPU
int gpu_maxthreads();

/// Overrides the maximum number of threads per block. max=0 means autoset.
void gpu_setmaxthreads(int max);

/**
 * Macro for 1D index "i" in a CUDA kernel.
 @code
  i = threadindex;
 @endcode
 */
#define threadindex ( ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x )

/**
 * @internal
 * Macro for integer division x/y , but rounded UP
 */
//#define divUp(x, y) ( (((x)-1)/(y)) +1 )
// It's almost like LISP ;-)
inline int divUp(int x, int y)
{
    long long xx = (long long)x;
    long long yy = (long long)y;
    long long res = ((xx - 1) / yy) + 1;
    return (int)res;
}

#ifndef X
#define X 0
#define Y 1
#define Z 2
#endif

/**
 * Checks if the CUDA 3D kernel launch configuration is valid.
 * CUDA tends to ignore invalid configurations silently, which is painfull for debugging.
 * Uses device properties
 */
///< 3D size of the thread grid
///< 3D size of the trhead blocks on the grid
void check3dconf(dim3 gridsize, dim3 blocksize );

/**
 * Checks if the CUDA 1D kernel launch configuration is valid.
 * CUDA tends to ignore invalid configurations silently, which is painfull for debugging.
 * Uses device properties
 */
void check1dconf(int gridsize, ///< 1D size of the thread grid
                 int blocksize ///< 1D size of the trhead blocks on the grid
                );


/**
 * Makes a 1D thread configuration suited for a float array of size N
 * The returned configuration will span at least the entire array but
 * can be larger. Your kernel should use the threadindex macro to
 * get the index "i", and check if it is smaller than the size "N" of
 * the array it is meant to iterate over.
 * @see make3dconf() threadindex
 *
 * Example:
 * @code
 * dim3 gridSize, blockSize;
 * make1dconf(arraySize, &gridSize, &blockSize);
 * mykernel<<<gridSize, blockSize>>>(arrrrghs, arraySize);

   __global__ void mykernel(int aargs, int N){
    int i = threadindex; //built-in macro
    if(i < N){  // check if i is in range!
      do_work();
    }
   }
 * @endcode
 */
void make1dconf(int N,           ///< size of array to span (number of floats)
                dim3* gridSize,  ///< grid size is returned here
                dim3* blockSize  ///< block size is returned here
               );


void make2dconf(int N1, int N2, dim3* gridSize, dim3* blockSize);

void make3dconf(int N0, int N1, int N2, dim3* gridSize, dim3* blockSize);

#ifdef __cplusplus
}
#endif
#endif
