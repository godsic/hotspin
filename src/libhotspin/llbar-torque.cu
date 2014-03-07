#include "llbar-torque.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "stdio.h"
#include <cuda.h>
#include "common_func.h"
#ifdef __cplusplus
extern "C" {
#endif

__global__ void llbarTorqueKern(float* __restrict__ tx, float* __restrict__ ty, float* __restrict__ tz,
                                float* __restrict__ Mx, float* __restrict__ My, float* __restrict__ Mz,
                                float* __restrict__ hx, float* __restrict__ hy, float* __restrict__ hz,

                                float* __restrict__ msat0T0Msk,

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
        float3 M = make_float3(Mx[x0], My[x0], Mz[x0]);

        float3 _MxH = crossf(H, M);

        tx[x0] = _MxH.x;
        ty[x0] = _MxH.y;
        tz[x0] = _MxH.z;
    }
}

#define BLOCKSIZE 16

__export__  void llbar_torque_async(float* tx, float*  ty, float*  tz,
                                    float*  Mx, float*  My, float*  Mz,
                                    float*  hx, float*  hy, float*  hz,

                                    float* msat0T0,

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
