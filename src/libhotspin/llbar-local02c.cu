#include "llbar-local02c.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "stdio.h"
#include <cuda.h>
#include "common_func.h"
#ifdef __cplusplus
extern "C" {
#endif

__global__ void llbarLocal02cKern(float* __restrict__ tx, float* __restrict__ ty, float* __restrict__ tz,
                                  float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
                                  float* __restrict__ hx, float* __restrict__ hy, float* __restrict__ hz,

                                  float* __restrict__ msat0T0Msk,

                                  float* __restrict__ mu_xx,
                                  float* __restrict__ mu_yy,
                                  float* __restrict__ mu_zz,

                                  const float muMul_xx,
                                  const float muMul_yy,
                                  const float muMul_zz,

                                  int Npart)
{

    int x0 = threadindex;

    if (x0 < Npart)
    {

        float msat0T0 = (msat0T0Msk == NULL) ? 1.0 : msat0T0Msk[x0];
        float3 m = make_float3(mx[x0], my[x0], mz[x0]);

        // make sure there is no torque for non-magnetic points
        if (msat0T0 == 0.0f)
        {
            tx[x0] = 0.0f;
            ty[x0] = 0.0f;
            tz[x0] = 0.0f;
            return;
        }

        float3 H = make_float3(hx[x0], hy[x0], hz[x0]);

        float3 mxH = crossf(m, H);

        float3 mu_mxH;

        float m_xx = (mu_xx != NULL) ? mu_xx[x0] * muMul_xx : muMul_xx;

        mu_mxH.x = m_xx * mxH.x;

        float m_yy = (mu_yy != NULL) ? mu_yy[x0] * muMul_yy : muMul_yy;

        mu_mxH.y = m_yy * mxH.y;

        float m_zz = (mu_zz != NULL) ? mu_zz[x0] * muMul_zz : muMul_zz;

        mu_mxH.z = m_zz * mxH.z;

        float3 _mxmu_mxH = crossf(mu_mxH, m);

        tx[x0] = _mxmu_mxH.x;
        ty[x0] = _mxmu_mxH.y;
        tz[x0] = _mxmu_mxH.z;
    }
}

__export__  void llbar_local02c_async(float* tx, float*  ty, float*  tz,
                                      float*  mx, float*  my, float*  mz,
                                      float*  hx, float*  hy, float*  hz,

                                      float* msat0T0,

                                      float* mu_xx,
                                      float* mu_yy,
                                      float* mu_zz,

                                      const float muMul_xx,
                                      const float muMul_yy,
                                      const float muMul_zz,

                                      CUstream stream,
                                      int Npart)
{
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);

    llbarLocal02cKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (tx, ty, tz,
            mx, my, mz,
            hx, hy, hz,

            msat0T0,

            mu_xx,
            mu_yy,
            mu_zz,

            muMul_xx,
            muMul_yy,
            muMul_zz,

            Npart);

}

// ========================================

#ifdef __cplusplus
}
#endif
