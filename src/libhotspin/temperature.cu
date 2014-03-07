
#include "temperature.h"

#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal
__global__ void temperature_scaleAnisKern(float* __restrict__ hx, float* __restrict__ hy, float* __restrict__ hz,

        float* __restrict__ mu_xx,
        float* __restrict__ mu_yy,
        float* __restrict__ mu_zz,

        float* __restrict__ tempMask,
        float* __restrict__ msat0T0Mask,

        const float muMul_xx,
        const float muMul_yy,
        const float muMul_zz,

        const float KB2tempMul_mu0VgammaDtMSatMul,

        int Npart)
{


    int i = threadindex;

    if (i < Npart)
    {

        float msat0T0 = getMaskUnity(msat0T0Mask, i);
        if (msat0T0 == 0.0f)
        {
            hx[i] = 0.0f;
            hy[i] = 0.0f;
            hz[i] = 0.0f;
            return;
        }

        float3 H = make_float3(hx[i], hy[i], hz[i]);

        float3 mu_H;

        float m_xx = muMul_xx * getMaskUnity(mu_xx, i);
        m_xx = sqrtf(m_xx);
        mu_H.x = m_xx * H.x;

        float m_yy = muMul_yy * getMaskUnity(mu_yy, i);
        m_yy = sqrtf(m_yy);
        mu_H.y = m_yy * H.y;

        float m_zz = muMul_zz * getMaskUnity(mu_zz, i);
        m_zz = sqrtf(m_zz);
        mu_H.z = m_zz * H.z;

        float T = getMaskUnity(tempMask, i);
        float pre = sqrtf((T * KB2tempMul_mu0VgammaDtMSatMul) / msat0T0);
        
        hx[i] = pre * mu_H.x;
        hy[i] = pre * mu_H.y;
        hz[i] = pre * mu_H.z;

    }
}


__export__ void temperature_scaleAnizNoise(float* hx, float* hy, float* hz,
        float* mu_xx,
        float* mu_yy,
        float* mu_zz,
        float* tempMask,
        float* msat0T0Mask,

        float muMul_xx,
        float muMul_yy,
        float muMul_zz,

        float KB2tempMul_mu0VgammaDtMSatMul,
        CUstream stream,
        int Npart)
{

    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    temperature_scaleAnisKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (
        hx, hy, hz,
        mu_xx,
        mu_yy,
        mu_zz,
        tempMask,
        msat0T0Mask,
        muMul_xx,
        muMul_yy,
        muMul_zz,
        KB2tempMul_mu0VgammaDtMSatMul,
        Npart);
}

#ifdef __cplusplus
}
#endif
