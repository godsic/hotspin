#include "div.h"

#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal
__global__ void divKern(double* dst, double* a, double* b, int Npart)
{
    int i = threadindex;
    if (i < Npart)
    {

        double bb = (b == NULL) ? 1.0 : b[i];
        dst[i] = (bb == 0.0) ? 0.0 : a[i] / bb;
    }
}


__export__ void divAsync(double* dst, double* a, double* b, CUstream stream, int Npart)
{
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    divKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (dst, a, b, Npart);
}

#ifdef __cplusplus
}
#endif

