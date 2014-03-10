#include "normalize.h"

#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal
__global__ void decomposeKern(double* __restrict__ Mx, double* __restrict__ My, double* __restrict__ Mz,
                              double* __restrict__ mx, double* __restrict__ my, double* __restrict__ mz,
                              double* __restrict__ msat,
                              double msatMul,
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
            mx[i] = 0.0;
            my[i] = 0.0;
            mz[i] = 0.0;
            msat[i] = 0.0;
            return;
        }

        mx[i] = (double)(M.x / Ms);
        my[i] = (double)(M.y / Ms);
        mz[i] = (double)(M.z / Ms);

        msat[i] = (double)Ms;
    }
}


__export__ void decomposeAsync(double* Mx, double* My, double* Mz,
                               double* mx, double* my, double* mz,
                               double* msat,
                               double msatMul,
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
