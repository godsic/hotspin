#include "normalize.h"

#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal
__global__ void decomposeKern(float* __restrict__ Mx, float* __restrict__ My, float* __restrict__ Mz,
                              float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
                              float* __restrict__ msat,
                              float msatMul,
                              int Npart)
{
    int i = threadindex;
    if (i < Npart)
    {

        // reconstruct norm from map

        double3 M = make_double3(Mx[i], My[i], Mz[i]);

        double Ms = len(M);

        if (Ms <= zero)
        {
            mx[i] = 0.0f;
            my[i] = 0.0f;
            mz[i] = 0.0f;
            msat[i] = 0.0f;
            return;
        }

        mx[i] = (float)(M.x / Ms);
        my[i] = (float)(M.y / Ms);
        mz[i] = (float)(M.z / Ms);

        msat[i] = (float)Ms;
    }
}


__export__ void decomposeAsync(float* Mx, float* My, float* Mz,
                               float* mx, float* my, float* mz,
                               float* msat,
                               float msatMul,
                               CUstream stream, int Npart)
{
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    decomposeKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (Mx, My, Mz,
            mx, my, mz,
            msat,
            msatMul,
            Npart);
}

#ifdef __cplusplus
}
#endif
