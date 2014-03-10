#include "llbar-local00nc.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "stdio.h"
#include <cuda.h>
#include "common_func.h"
#ifdef __cplusplus
extern "C" {
#endif

__global__ void llbarLocal00ncKern(double* __restrict__ tx, double* __restrict__ ty, double* __restrict__ tz,
                                   double* __restrict__ hx, double* __restrict__ hy, double* __restrict__ hz,

                                   double* __restrict__ msat0T0Msk,

                                   double* __restrict__ lambda_xx,
                                   double* __restrict__ lambda_yy,
                                   double* __restrict__ lambda_zz,

                                   const double lambdaMul_xx,
                                   const double lambdaMul_yy,
                                   const double lambdaMul_zz,

                                   int Npart)
{

    int x0 = threadindex;

    if (x0 < Npart)
    {

        double msat0T0 = (msat0T0Msk == NULL) ? 1.0 : msat0T0Msk[x0];

        // make sure there is no torque in vacuum!
        if (msat0T0 == 0.0)
        {
            tx[x0] = 0.0;
            ty[x0] = 0.0;
            tz[x0] = 0.0;
            return;
        }

        double3 H = make_double3(hx[x0], hy[x0], hz[x0]);

        double3 lambda_H;

        double l_xx = (lambda_xx != NULL) ? lambda_xx[x0] * lambdaMul_xx : lambdaMul_xx;

        lambda_H.x = l_xx * H.x;

        double l_yy = (lambda_yy != NULL) ? lambda_yy[x0] * lambdaMul_yy : lambdaMul_yy;

        lambda_H.y = l_yy * H.y;

        double l_zz = (lambda_zz != NULL) ? lambda_zz[x0] * lambdaMul_zz : lambdaMul_zz;

        lambda_H.z = l_zz * H.z;

        tx[x0] = lambda_H.x;
        ty[x0] = lambda_H.y;
        tz[x0] = lambda_H.z;
    }
}

__export__  void llbar_local00nc_async(double* tx, double*  ty, double*  tz,
                                       double*  hx, double*  hy, double*  hz,

                                       double* msat0T0,

                                       double* lambda_xx,
                                       double* lambda_yy,
                                       double* lambda_zz,

                                       const double lambdaMul_xx,
                                       const double lambdaMul_yy,
                                       const double lambdaMul_zz,

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
