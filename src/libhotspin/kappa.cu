#include "kappa.h"

#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal


__global__ void kappaKern(double* __restrict__ kappa,
                          double* __restrict__ msat0Msk,
                          double* __restrict__ msat0T0Msk,
                          double* __restrict__ T,
                          double* __restrict__ TcMsk,
                          double* __restrict__ SMsk,
                          double* __restrict__ nMsk,
                          const double msat0Mul,
                          const double msat0T0Mul,
                          const double TcMul,
                          const double SMul,
                          int Npart)
{

    int i = threadindex;

    if (i < Npart)
    {
        double msat0T0u = getMaskUnity(msat0T0Msk, i);
        double msat0T0 = msat0T0Mul * msat0T0u;

        double Temp = T[i];

        if (msat0T0 == 0.0 || Temp == 0.0)
        {
            kappa[i] = 0.0;
            return;
        }


        double S = SMul * getMaskUnity(SMsk, i);
        double Tc = TcMul * getMaskUnity(TcMsk, i);
        double msat0 = msat0Mul * getMaskUnity(msat0Msk, i);
        double preS = (S > INFINITESPINLIMIT) ? 1.0 : S / (S + 1.0);
        double J0  = 3.0 * Tc;
        double n = (nMsk == NULL) ? 1.0 : nMsk[i];

        double mul = msat0T0u * msat0T0u / (preS * J0 * n); // msat0T0 mul should be in the kappa multiplier
        double me = msat0 / msat0T0;
        double b = preS * J0 / Temp;
        double meb = me * b;
        double f = (S > INFINITESPINLIMIT) ? dLdx(meb) : dBjdx(S, meb);
        f = b * f;
        double k = mul * (f / (1.0 - f));
        kappa[i] = k;
    }
}

__export__ void kappaAsync(double* kappa,
                           double* msat0,
                           double* msat0T0,
                           double* T,
                           double* Tc,
                           double* S,
                           double* n,
                           const double msat0Mul,
                           const double msat0T0Mul,
                           const double TcMul,
                           const double SMul,
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
