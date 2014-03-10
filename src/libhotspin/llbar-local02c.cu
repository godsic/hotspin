#include "llbar-local02c.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "stdio.h"
#include <cuda.h>
#include "common_func.h"
#ifdef __cplusplus
extern "C" {
#endif

__global__ void llbarLocal02cKern(double* __restrict__ tx, double* __restrict__ ty, double* __restrict__ tz,
                                  double* __restrict__ mx, double* __restrict__ my, double* __restrict__ mz,
                                  double* __restrict__ hx, double* __restrict__ hy, double* __restrict__ hz,

                                  double* __restrict__ msat0T0Msk,

                                  double* __restrict__ mu_xx,
                                  double* __restrict__ mu_yy,
                                  double* __restrict__ mu_zz,

                                  const double muMul_xx,
                                  const double muMul_yy,
                                  const double muMul_zz,

                                  int Npart)
{

    int x0 = threadindex;

    if (x0 < Npart)
    {

        double msat0T0 = (msat0T0Msk == NULL) ? 1.0 : msat0T0Msk[x0];
        double3 m = make_double3(mx[x0], my[x0], mz[x0]);

        // make sure there is no torque for non-magnetic points
        if (msat0T0 == 0.0)
        {
            tx[x0] = 0.0;
            ty[x0] = 0.0;
            tz[x0] = 0.0;
            return;
        }

        double3 H = make_double3(hx[x0], hy[x0], hz[x0]);

        double3 mxH = cross(m, H);

        double3 mu_mxH;

        double m_xx = (mu_xx != NULL) ? mu_xx[x0] * muMul_xx : muMul_xx;

        mu_mxH.x = m_xx * mxH.x;

        double m_yy = (mu_yy != NULL) ? mu_yy[x0] * muMul_yy : muMul_yy;

        mu_mxH.y = m_yy * mxH.y;

        double m_zz = (mu_zz != NULL) ? mu_zz[x0] * muMul_zz : muMul_zz;

        mu_mxH.z = m_zz * mxH.z;

        double3 _mxmu_mxH = cross(mu_mxH, m);

        tx[x0] = _mxmu_mxH.x;
        ty[x0] = _mxmu_mxH.y;
        tz[x0] = _mxmu_mxH.z;
    }
}

__export__  void llbar_local02c_async(double* tx, double*  ty, double*  tz,
                                      double*  mx, double*  my, double*  mz,
                                      double*  hx, double*  hy, double*  hz,

                                      double* msat0T0,

                                      double* mu_xx,
                                      double* mu_yy,
                                      double* mu_zz,

                                      const double muMul_xx,
                                      const double muMul_yy,
                                      const double muMul_zz,

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
