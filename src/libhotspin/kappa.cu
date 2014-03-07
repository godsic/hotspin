#include "kappa.h"

#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal


__global__ void kappaKern(float* __restrict__ kappa,
                          float* __restrict__ msat0Msk,
                          float* __restrict__ msat0T0Msk,
                          float* __restrict__ T,
                          float* __restrict__ TcMsk,
                          float* __restrict__ SMsk,
                          float* __restrict__ nMsk,
                          const float msat0Mul,
                          const float msat0T0Mul,
                          const float TcMul,
                          const float SMul,
                          int Npart)
{

    int i = threadindex;

    if (i < Npart)
    {
        double msat0T0u = (msat0T0Msk == NULL) ? 1.0 : msat0T0Msk[i];
        double msat0T0 = msat0T0Mul * msat0T0u;

        double Temp = T[i];

        if (msat0T0 == 0.0 || Temp == 0.0)
        {
            kappa[i] = 0.0f;
            return;
        }


        double S = (SMsk == NULL) ? SMul : SMul * SMsk[i];
        double Tc = (TcMsk == NULL) ? TcMul : TcMul * TcMsk[i];
        double msat0 = (msat0Msk == NULL) ? msat0Mul : msat0Mul * msat0Msk[i];
        double J0  = 3.0 * Tc / (S * (S + 1.0)); // in h^2 units
        double n = (nMsk == NULL) ? 1.0 : nMsk[i];

        double mul = msat0T0u * msat0T0u / (S * S * J0 * n); // msat0T0 mul should be in the kappa multiplier
        double me = msat0 / msat0T0;
        double b = S * S * J0 / Temp;
        double meb = me * b;
        double f = b * dBjdx(S, meb);
        double k = mul * (f / (1.0 - f));
        kappa[i] = (float)k;
    }
}

__export__ void kappaAsync(float* kappa,
                           float* msat0,
                           float* msat0T0,
                           float* T,
                           float* Tc,
                           float* S,
                           float* n,
                           const float msat0Mul,
                           const float msat0T0Mul,
                           const float TcMul,
                           const float SMul,
                           int Npart,
                           CUstream stream)
{
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    kappaKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (kappa,
            msat0,
            msat0T0,
            T,
            Tc,
            S,
            n,
            msat0Mul,
            msat0T0Mul,
            TcMul,
            SMul,
            Npart);
}

#ifdef __cplusplus
}
#endif
