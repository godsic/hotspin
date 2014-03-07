#include "llbar-local00nc.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "stdio.h"
#include <cuda.h>
#include "common_func.h"
#ifdef __cplusplus
extern "C" {
#endif

__global__ void llbarLocal00ncKern(float* __restrict__ tx, float* __restrict__ ty, float* __restrict__ tz,
                                   float* __restrict__ hx, float* __restrict__ hy, float* __restrict__ hz,

                                   float* __restrict__ msat0T0Msk,

                                   float* __restrict__ lambda_xx,
                                   float* __restrict__ lambda_yy,
                                   float* __restrict__ lambda_zz,

                                   const float lambdaMul_xx,
                                   const float lambdaMul_yy,
                                   const float lambdaMul_zz,

                                   int Npart)
{

    int x0 = threadindex;

    if (x0 < Npart)
    {

        float msat0T0 = (msat0T0Msk == NULL) ? 1.0 : msat0T0Msk[x0];

        // make sure there is no torque in vacuum!
        if (msat0T0 == 0.0f)
        {
            tx[x0] = 0.0f;
            ty[x0] = 0.0f;
            tz[x0] = 0.0f;
            return;
        }

        float3 H = make_float3(hx[x0], hy[x0], hz[x0]);

        float3 lambda_H;

        float l_xx = (lambda_xx != NULL) ? lambda_xx[x0] * lambdaMul_xx : lambdaMul_xx;

        lambda_H.x = l_xx * H.x;

        float l_yy = (lambda_yy != NULL) ? lambda_yy[x0] * lambdaMul_yy : lambdaMul_yy;

        lambda_H.y = l_yy * H.y;

        float l_zz = (lambda_zz != NULL) ? lambda_zz[x0] * lambdaMul_zz : lambdaMul_zz;

        lambda_H.z = l_zz * H.z;

        tx[x0] = lambda_H.x;
        ty[x0] = lambda_H.y;
        tz[x0] = lambda_H.z;
    }
}

__export__  void llbar_local00nc_async(float* tx, float*  ty, float*  tz,
                                       float*  hx, float*  hy, float*  hz,

                                       float* msat0T0,

                                       float* lambda_xx,
                                       float* lambda_yy,
                                       float* lambda_zz,

                                       const float lambdaMul_xx,
                                       const float lambdaMul_yy,
                                       const float lambdaMul_zz,

                                       CUstream stream,
                                       int Npart)
{
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);

    llbarLocal00ncKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (tx, ty, tz,
            hx, hy, hz,

            msat0T0,

            lambda_xx,
            lambda_yy,
            lambda_zz,

            lambdaMul_xx,
            lambdaMul_yy,
            lambdaMul_zz,

            Npart);

}

// ========================================

#ifdef __cplusplus
}
#endif
