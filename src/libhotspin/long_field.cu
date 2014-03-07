#include "long_field.h"
#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"


#ifdef __cplusplus
extern "C" {
#endif
// ========================================

__global__ void long_field_Kern(float* __restrict__ hx, float* __restrict__ hy, float* __restrict__ hz,
                                float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
                                float* __restrict__ msat0T0Msk,
                                float* __restrict__ SMsk,
                                float* __restrict__ nMsk,
                                float* __restrict__ TcMsk,
                                float* __restrict__ TsMsk,
                                float msat0T0Mul,
                                float SMul,
                                float nMul,
                                float TcMul,
                                float TsMul,
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
            hx[I] = 0.0f;
            hy[I] = 0.0f;
            hz[I] = 0.0f;
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


__export__ void long_field_async(float* hx, float* hy, float* hz,
                                 float* mx, float* my, float* mz,
                                 float* msat0T0,
                                 float* S,
                                 float* n,
                                 float* Tc,
                                 float* Ts,
                                 float msat0T0Mul,
                                 float SMul,
                                 float nMul,
                                 float TcMul,
                                 float TsMul,
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
