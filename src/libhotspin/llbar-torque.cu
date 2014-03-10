#include "llbar-torque.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "stdio.h"
#include <cuda.h>
#include "common_func.h"
#ifdef __cplusplus
extern "C" {
#endif

__global__ void llbarTorqueKern(double* __restrict__ tx, double* __restrict__ ty, double* __restrict__ tz,
                                double* __restrict__ Mx, double* __restrict__ My, double* __restrict__ Mz,
                                double* __restrict__ hx, double* __restrict__ hy, double* __restrict__ hz,

                                double* __restrict__ msat0T0Msk,

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
        double3 M = make_double3(Mx[x0], My[x0], Mz[x0]);

        double3 _MxH = cross(H, M);

        tx[x0] = _MxH.x;
        ty[x0] = _MxH.y;
        tz[x0] = _MxH.z;
    }
}

#define BLOCKSIZE 16

__export__  void llbar_torque_async(double* tx, double*  ty, double*  tz,
                                    double*  Mx, double*  My, double*  Mz,
                                    double*  hx, double*  hy, double*  hz,

                                    double* msat0T0,

                                    CUstream stream,
                                    int Npart)
{
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    llbarTorqueKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (tx, ty, tz,
            Mx, My, Mz,
            hx, hy, hz,

            msat0T0,

            Npart);

}

// ========================================

#ifdef __cplusplus
}
#endif
