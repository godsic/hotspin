#include "wavg.h"

#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal
__global__ void wavgKern(double* dst, 
                         double* a, double* b, 
                         double* w0Msk, double* w1Msk, 
                         double* RMsk,
                         double w0Mul,
                         double w1Mul,
                         double RMul,
                         int Npart)
{
    int i = threadindex;
    if (i < Npart)
    {
        double w0 = w0Mul * getMaskUnity(w0Msk, i);
        double w1 = w1Mul * getMaskUnity(w1Msk, i);
        double R = RMul * getMaskUnity(RMsk, i);
        dst[i] = weightedAvgZero(a[i], b[i], w0, w1, R);
    }
}


__export__ void wavgAsync(double* dst, 
                         double* a, double* b, 
                         double* w0, double* w1, 
                         double* R,
                         double w0Mul,
                         double w1Mul,
                         double RMul,
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

