
#include "temperature.h"

#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal
__global__ void temperature_scaleAnisKern(double* __restrict__ hx, double* __restrict__ hy, double* __restrict__ hz,

        double* __restrict__ mu_xx,
        double* __restrict__ mu_yy,
        double* __restrict__ mu_zz,

        double* __restrict__ tempMask,
        double* __restrict__ msat0T0Mask,

        const double muMul_xx,
        const double muMul_yy,
        const double muMul_zz,

        const double KB2tempMul_mu0VgammaDtMSatMul,

        int Npart)
{


    int i = threadindex;

    if (i < Npart)
    {

        double msat0T0 = getMaskUnity(msat0T0Mask, i);
        if (msat0T0 == 0.0)
        {
            hx[i] = 0.0;
            hy[i] = 0.0;
            hz[i] = 0.0;
            return;
        }

        double3 H = make_double3(hx[i], hy[i], hz[i]);

        double3 mu_H;

        double m_xx = muMul_xx * getMaskUnity(mu_xx, i);
        m_xx = sqrt(m_xx);
        mu_H.x = m_xx * H.x;

        double m_yy = muMul_yy * getMaskUnity(mu_yy, i);
        m_yy = sqrt(m_yy);
        mu_H.y = m_yy * H.y;

        double m_zz = muMul_zz * getMaskUnity(mu_zz, i);
        m_zz = sqrt(m_zz);
        mu_H.z = m_zz * H.z;

        double T = getMaskUnity(tempMask, i);
        double pre = sqrt((T * KB2tempMul_mu0VgammaDtMSatMul) / msat0T0);
        
        hx[i] = pre * mu_H.x;
        hy[i] = pre * mu_H.y;
        hz[i] = pre * mu_H.z;

    }
}


__export__ void temperature_scaleAnizNoise(double* hx, double* hy, double* hz,
        double* mu_xx,
        double* mu_yy,
        double* mu_zz,
        double* tempMask,
        double* msat0T0Mask,

        double muMul_xx,
        double muMul_yy,
        double muMul_zz,

        double KB2tempMul_mu0VgammaDtMSatMul,
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
