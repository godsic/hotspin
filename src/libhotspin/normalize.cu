#include "normalize.h"

#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal
__global__ void normalizeKern(float* mx, float* my, float* mz,
                              int Npart)
{
    int i = threadindex;

    if (i < Npart)
    {
        // reconstruct norm from map

        float Mx = mx[i];
        float My = my[i];
        float Mz = mz[i];

        float Mnorm = 1.0f / sqrtf(Mx * Mx + My * My + Mz * Mz);

        mx[i] = Mx * Mnorm;
        my[i] = My * Mnorm;
        mz[i] = Mz * Mnorm;
    }
}


__export__ void normalizeAsync(float* mx, float* my, float* mz, CUstream stream, int Npart)
{
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    normalizeKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (mx, my, mz, Npart);
}

#ifdef __cplusplus
}
#endif
