#include "energy_flow.h"

#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal


__global__ void energyFlowKern(float* __restrict__ w,
                          float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz, 
                          float* __restrict__ Rx, float* __restrict__ Ry, float* __restrict__ Rz,
                          float* __restrict__ TcMsk,
                          float* __restrict__ SMsk,
                          float* __restrict__ nMsk,
                          float SMul,
                          int Npart)
{

    int i = threadindex;

    if (i < Npart)
    {
        float S = SMul * getMaskUnity(SMsk, i);
        float Tc = getMaskUnity(TcMsk, i);
        float n = getMaskUnity(nMsk, i);

        if (n == 0.0f || S == 0.0f  || Tc == 0.0f)
        {
            w[i] = 0.0f;
            return;
        }

        float3 m = make_float3(mx[i], my[i], mz[i]);
        float3 R = make_float3(Rx[i], Ry[i], Rz[i]);
        
        float rW = dotf(m, R);

        float mult = Tc * S * S * n / (S * (S + 1.0f));

        w[i] = mult * rW;
    }
}

__export__ void energyFlowAsync(float* w,
                          float* mx, float* my, float* mz,
                          float* Rx, float* Ry, float* Rz,
                          float* Tc,
                          float* S,
                          float* n,
                          float SMul,
                          int Npart,
                          CUstream stream)
{
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    energyFlowKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (w,
            mx, my, mz,
            Rx, Ry, Rz,
            Tc,
            S,
            n,
            SMul,
            Npart);
}

#ifdef __cplusplus
}
#endif
