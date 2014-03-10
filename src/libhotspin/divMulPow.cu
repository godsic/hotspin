#include "div.h"

#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal
__global__ void divMulPowKern(double* __restrict__ dst,
                              double* __restrict__ a,
                              double* __restrict__ b,
                              double* __restrict__ c,
                              double p, int Npart)
{
    int i = threadindex;
    if (i < Npart)
    {

        double bb = (b == NULL) ? 1.0 : b[i];
        double val = (bb == 0.0) ? 0.0 : a[i] / bb;

        double cc = (c == NULL) ? 1.0 : c[i];
        cc = (cc == 0.0) ? 0.0 : powf(cc, p);

        dst[i] = val * cc;
    }
}


__export__ void divMulPowAsync(double* dst, double* a, double* b, double* c, double p, CUstream stream, int Npart)
{
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    divMulPowKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (dst, a, b, c, p, Npart);
}

#ifdef __cplusplus
}
#endif

