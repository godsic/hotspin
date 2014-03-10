#include "energy_flow.h"

#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal


__global__ void energyFlowKern(double* __restrict__ w,
                          double* __restrict__ mx, double* __restrict__ my, double* __restrict__ mz, 
                          double* __restrict__ Rx, double* __restrict__ Ry, double* __restrict__ Rz,
                          double* __restrict__ TcMsk,
                          double* __restrict__ SMsk,
                          double* __restrict__ nMsk,
                          double SMul,
                          int Npart)
{

    int i = threadindex;

    if (i < Npart)
    {
        double S = SMul * getMaskUnity(SMsk, i);
        double Tc = getMaskUnity(TcMsk, i);
        double n = getMaskUnity(nMsk, i);

        if (n == 0.0 || S == 0.0  || Tc == 0.0)
        {
            w[i] = 0.0;
            return;
        }

        double3 m = make_double3(mx[i], my[i], mz[i]);
        double3 R = make_double3(Rx[i], Ry[i], Rz[i]);
        
        double rW = dot(m, R);

        double mult = Tc * S * S * n / (S * (S + 1.0));

        w[i] = mult * rW;
    }
}

__export__ void energyFlowAsync(double* w,
                          double* mx, double* my, double* mz,
                          double* Rx, double* Ry, double* Rz,
                          double* Tc,
                          double* S,
                          double* n,
                          double SMul,
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
