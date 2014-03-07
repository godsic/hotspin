#include "div.h"

#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal
__global__ void divMulPowKern(float* __restrict__ dst,
                              float* __restrict__ a,
                              float* __restrict__ b,
                              float* __restrict__ c,
                              float p, int Npart)
{
    int i = threadindex;
    if (i < Npart)
    {

        float bb = (b == NULL) ? 1.0f : b[i];
        float val = (bb == 0.0f) ? 0.0f : a[i] / bb;

        float cc = (c == NULL) ? 1.0f : c[i];
        cc = (cc == 0.0) ? 0.0 : powf(cc, p);

        dst[i] = val * cc;
    }
}


__export__ void divMulPowAsync(float* dst, float* a, float* b, float* c, float p, CUstream stream, int Npart)
{
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    divMulPowKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (dst, a, b, c, p, Npart);
}

#ifdef __cplusplus
}
#endif

