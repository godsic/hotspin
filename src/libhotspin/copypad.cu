#include "copypad.h"

#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"

#ifdef __cplusplus
extern "C" {
#endif


__global__ void zeroArrayKern(double *A, int N)
{

    int i = threadindex;

    if (i < N)
    {
        A[i] = 0.0;
    }
}

__export__ void zeroArrayAsync(double *A, int length, CUstream streams)
{

    dim3 gridSize, blockSize;
    make1dconf(length, &gridSize, &blockSize);
    zeroArrayKern <<< gridSize, blockSize, 0, cudaStream_t(streams)>>>( A, length );
}


/// @internal Does padding and unpadding of a 3D matrix.  Padding in the y-direction is only correct when 1 GPU is used!!
/// Fills padding space with zeros.
__global__ void copyPad3DKern(double* __restrict__ dst, int D0, int D1, int D2, int D1D2, double* __restrict__ src, int S0, int S1, int S2, int S1S2)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // this check makes it work for padding as well as for unpadding.
    // 2 separate functions are probably not more efficient
    // due to memory bandwidth limitations
    if (i < S0 && j < S1 && k < S2) // if we are in the destination array we should write something
    {
        dst[i * D1D2 + j * D2 + k] = src[i * S1S2 + j * S2 + k];
    }
}

__export__ void copyPad3DAsync(double* dst, int D0, int D1, int D2, double* src, int S0, int S1, int S2, int Ncomp, CUstream streams)
{

    dim3 gridSize, blockSize;
    make3dconf(S0, S1, S2, &gridSize, &blockSize);

    int D1D2 = D1 * D2;
    int S1S2 = S1 * S2;
    for (int i = 0; i < Ncomp; i++)
    {
        double* src3D = &(src[i * S0 * S1S2]);
        double* dst3D = &(dst[i * D0 * D1D2]); //D1==S1
        copyPad3DKern <<< gridSize, blockSize, 0, cudaStream_t(streams)>>> (dst3D, D0, D1, D2, D1D2, src3D, S0, S1, S2, S1S2);
    }
}


__global__ void copyUnPad3DKern(double* __restrict__ dst, int D0, int D1, int D2, int D1D2, double* __restrict__ src, int S0, int S1, int S2, int S1S2)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // this check makes it work for padding as well as for unpadding.
    // 2 separate functions are probably not more efficient
    // due to memory bandwidth limitations
    if (i < D0 && j < D1 && k < D2) // if we are in the destination array we should write something
    {
        dst[i * D1D2 + j * D2 + k] = src[i * S1S2 + j * S2 + k];
    }
}

__export__ void copyUnPad3DAsync(double* dst, int D0, int D1, int D2, double* src, int S0, int S1, int S2, int Ncomp, CUstream streams)
{

    dim3 gridSize, blockSize;
    make3dconf(D0, D1, D2, &gridSize, &blockSize);

    int D1D2 = D1 * D2;
    int S1S2 = S1 * S2;
    for (int i = 0; i < Ncomp; i++)
    {
        double* src3D = &(src[i * S0 * S1S2]);
        double* dst3D = &(dst[i * D0 * D1D2]); //D1==S1
        copyUnPad3DKern <<< gridSize, blockSize, 0, cudaStream_t(streams)>>> (dst3D, D0, D1, D2, D1D2, src3D, S0, S1, S2, S1S2);
    }
}


#ifdef __cplusplus
}
#endif

