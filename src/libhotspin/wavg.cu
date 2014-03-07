#include "wavg.h"

#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal
__global__ void wavgKern(float* dst, 
                         float* a, float* b, 
                         float* w0Msk, float* w1Msk, 
                         float* RMsk,
                         float w0Mul,
                         float w1Mul,
                         float RMul,
                         int Npart)
{
    int i = threadindex;
    if (i < Npart)
    {
        float w0 = w0Mul * getMaskUnity(w0Msk, i);
        float w1 = w1Mul * getMaskUnity(w1Msk, i);
        float R = RMul * getMaskUnity(RMsk, i);
        dst[i] = weightedAvgZero(a[i], b[i], w0, w1, R);
    }
}


__export__ void wavgAsync(float* dst, 
                         float* a, float* b, 
                         float* w0, float* w1, 
                         float* R,
                         float w0Mul,
                         float w1Mul,
                         float RMul,
                         CUstream stream, int Npart)
{
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    wavgKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (dst, 
                                                                       a, b, 
                                                                       w0, w1,
                                                                       R,
                                                                       w0Mul, w1Mul,
                                                                       RMul,
                                                                       Npart);
}

#ifdef __cplusplus
}
#endif

