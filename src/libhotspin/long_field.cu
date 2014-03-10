#include "long_field.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"


#ifdef __cplusplus
extern "C" {
#endif
// ========================================

__global__ void long_field_Kern(double* __restrict__ hx, double* __restrict__ hy, double* __restrict__ hz,
                                double* __restrict__ mx, double* __restrict__ my, double* __restrict__ mz,
                                double* __restrict__ msat0T0Msk,
                                double* __restrict__ SMsk,
                                double* __restrict__ nMsk,
                                double* __restrict__ TcMsk,
                                double* __restrict__ TsMsk,
                                double msat0T0Mul,
                                double SMul,
                                double nMul,
                                double TcMul,
                                double TsMul,
                                int NPart)
{

    int I = threadindex;

    if (I < NPart)  // Thread configurations are usually too large...
    {

        double Ms0T0 = msat0T0Mul * getMaskUnity(msat0T0Msk, I);
        double S = SMul * getMaskUnity(SMsk, I);
        double n = nMul * getMaskUnity(nMsk, I);
        double Tc = TcMul * getMaskUnity(TcMsk, I);
        double Ts = TsMul * getMaskUnity(TsMsk, I);
        double3 mf = make_double3(mx[I], my[I], mz[I]);
        double abs_mf = len(mf);

        if (Ms0T0 == 0.0 || n == 0.0 || abs_mf <= zero)
        {
            hx[I] = 0.0;
            hy[I] = 0.0;
            hz[I] = 0.0;
            return;
        }
        
        double3 s = normalize(mf);

        double J0  = 3.0 * Tc / (S * (S + 1.0));

        double b = S * S * J0 / Ts;

        double meb = abs_mf * b;

        double M = Ms0T0 * abs_mf;

        double M0 = Ms0T0 * Bj(S, meb);

        double mult = n * kB * Ts * (M0 - M) / (mu0 * Ms0T0 * Ms0T0 * dBjdx(S, meb));

        hx[I] = mult * s.x;
        hy[I] = mult * s.y;
        hz[I] = mult * s.z;

    }
}


__export__ void long_field_async(double* hx, double* hy, double* hz,
                                 double* mx, double* my, double* mz,
                                 double* msat0T0,
                                 double* S,
                                 double* n,
                                 double* Tc,
                                 double* Ts,
                                 double msat0T0Mul,
                                 double SMul,
                                 double nMul,
                                 double TcMul,
                                 double TsMul,
                                 int NPart,
                                 CUstream stream)
{
    dim3 gridSize, blockSize;
    make1dconf(NPart, &gridSize, &blockSize);
    long_field_Kern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (hx, hy, hz,
            mx, my, mz,
            msat0T0,
            S,
            n,
            Tc,
            Ts,
            msat0T0Mul,
            SMul,
            nMul,
            TcMul,
            TsMul,
            NPart);
}

// ========================================

#ifdef __cplusplus
}
#endif
