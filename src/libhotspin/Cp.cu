#include "Cp.h"

#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal

__global__ void cpKern(float* __restrict__ Cp,
                          float* __restrict__ Tp,
                          float* __restrict__ TdMsk,
                          float* __restrict__ nMsk,
                          const float TdMul,
                          int Npart)
{

    int i = threadindex;

    if (i < Npart)
    {
        double n = getMaskUnity(nMsk, i);
        double Td = TdMul * getMaskUnity(TdMsk, i);
        double T = Tp[i];

        if (T == 0.0f || Td == 0.0f || n == 0.0f)
        {
            Cp[i] = 0.0f;
            return;
        }

        float xx = Td / T;

        float h = xx / (float)INTMAXSTEPS;
        float h_2 = 0.5f * h;

        float x = 0.0;  
        float val = 0.0f;

        while (x < xx) {
            val += (h_2 * (Debye(x) + Debye(x+h)));
            x += h;
        }

//         int i = 0;

// #pragma unroll 2
//         for (i = 0; i < INTMAXSTEPS; i++) {
//             val += (Debye(x) * h);
//             x += h;
//         }

        Cp[i] = 9.0f * n * val / (xx * xx * xx); // kb, nMul should be accounted in the upstream multiplier

    }
}

__export__ void cpAsync(float* Cp,
                          float* T,
                          float* Td,
                          float* n,
                          const float TdMul,
                          int Npart,
                          CUstream stream)
{
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    cpKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (Cp,
            T,
            Td,
            n,
            TdMul,
            Npart);

}

#ifdef __cplusplus
}
#endif
