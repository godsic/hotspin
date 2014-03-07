#include "llbar-nonlocal00nc.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "stdio.h"
#include <cuda.h>
#include "common_func.h"
#ifdef __cplusplus
extern "C" {
#endif

__global__ void llbarNonlocal00ncKern(float* __restrict__ tx, float* __restrict__ ty, float* __restrict__ tz,

                                      float* __restrict__ Hx, float* __restrict__ Hy, float* __restrict__ Hz,

                                      float* __restrict__ msat0T0Msk,

                                      float* __restrict__ lambda_e_xx,
                                      float* __restrict__ lambda_e_yy,
                                      float* __restrict__ lambda_e_zz,

                                      const float lambda_eMul_xx,
                                      const float lambda_eMul_yy,
                                      const float lambda_eMul_zz,

                                      const int3 N,
                                      const float3 cell_2,
                                      const int3 wrap)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < N.x && j < N.y && k < N.z)
    {

        int I = i * N.y * N.z + j * N.z + k;

        float msat0T0 = getMaskUnity(msat0T0Msk, I);
        // make sure there is no damping in vacuum!
        if (msat0T0 == 0.0f)
        {
            tx[I] = 0.0f;
            ty[I] = 0.0f;
            tz[I] = 0.0f;
            return;
        }

        // Second-order derivative 3-points stencil
//==================================================================================================

        float lexx0 = lambda_eMul_xx * getMaskUnity(lambda_e_xx, I);
        float leyy0 = lambda_eMul_yy * getMaskUnity(lambda_e_yy, I);
        float lezz0 = lambda_eMul_zz * getMaskUnity(lambda_e_zz, I);

        float lexx, leyy, lezz;
        float lexx1, leyy1, lezz1;
        float lexx2, leyy2, lezz2;

        float Hx0 = Hx[I]; // mag component of central cell
        float Hx1, Hx2;

        float Hy0 = Hy[I]; // mag component of central cell
        float Hy1, Hy2;

        float Hz0 = Hz[I]; // mag component of central cell
        float Hz1, Hz2;

        float Rx, Ry, Rz;

        int linAddr;

        // neighbors in X direction
        int idx = i - 1;
        idx = (idx < 0 && wrap.x) ? N.x + idx : idx;
        idx = max(idx, 0);
        linAddr = idx * N.y * N.z + j * N.z + k;

        lexx = lambda_eMul_xx * getMaskUnity(lambda_e_xx, linAddr);
        leyy = lambda_eMul_yy * getMaskUnity(lambda_e_yy, linAddr);
        lezz = lambda_eMul_zz * getMaskUnity(lambda_e_zz, linAddr);

        lexx1 = avgGeomZero(lexx0, lexx);
        leyy1 = avgGeomZero(leyy0, leyy);
        lezz1 = avgGeomZero(lezz0, lezz);

        Hx1 = Hx[linAddr];
        Hy1 = Hy[linAddr];
        Hz1 = Hz[linAddr];

        idx = i + 1;
        idx = (idx == N.x && wrap.x) ? idx - N.x : idx;
        idx = min(idx, N.x - 1);
        linAddr = idx * N.y * N.z + j * N.z + k;

        lexx = lambda_eMul_xx * getMaskUnity(lambda_e_xx, linAddr);
        leyy = lambda_eMul_yy * getMaskUnity(lambda_e_yy, linAddr);
        lezz = lambda_eMul_zz * getMaskUnity(lambda_e_zz, linAddr);

        lexx2 = avgGeomZero(lexx0, lexx);
        leyy2 = avgGeomZero(leyy0, leyy);
        lezz2 = avgGeomZero(lezz0, lezz);

        Hx2 = Hx[linAddr];
        Hy2 = Hy[linAddr];
        Hz2 = Hz[linAddr];

        Rx = cell_2.x * (lexx1 * (Hx1 - Hx0) + lexx2 * (Hx2 -  Hx0));
        Ry = cell_2.y * (leyy1 * (Hy1 - Hy0) + leyy2 * (Hy2 -  Hy0));
        Rz = cell_2.z * (lezz1 * (Hz1 - Hz0) + lezz2 * (Hz2 -  Hz0));

        // neighbors in Z direction
        idx = k - 1;
        idx = (idx < 0 && wrap.z) ? N.z + idx : idx;
        idx = max(idx, 0);
        linAddr = i * N.y * N.z + j * N.z + idx;

        lexx = lambda_eMul_xx * getMaskUnity(lambda_e_xx, linAddr);
        leyy = lambda_eMul_yy * getMaskUnity(lambda_e_yy, linAddr);
        lezz = lambda_eMul_zz * getMaskUnity(lambda_e_zz, linAddr);

        lexx1 = avgGeomZero(lexx0, lexx);
        leyy1 = avgGeomZero(leyy0, leyy);
        lezz1 = avgGeomZero(lezz0, lezz);

        Hx1 = Hx[linAddr];
        Hy1 = Hy[linAddr];
        Hz1 = Hz[linAddr];

        idx = k + 1;
        idx = (idx == N.z && wrap.z) ? idx - N.z : idx;
        idx = min(idx, N.z - 1);
        linAddr = i * N.y * N.z + j * N.z + idx;

        lexx = lambda_eMul_xx * getMaskUnity(lambda_e_xx, linAddr);
        leyy = lambda_eMul_yy * getMaskUnity(lambda_e_yy, linAddr);
        lezz = lambda_eMul_zz * getMaskUnity(lambda_e_zz, linAddr);

        lexx2 = avgGeomZero(lexx0, lexx);
        leyy2 = avgGeomZero(leyy0, leyy);
        lezz2 = avgGeomZero(lezz0, lezz);

        Hx2 = Hx[linAddr];
        Hy2 = Hy[linAddr];
        Hz2 = Hz[linAddr];

        Rx += cell_2.x * (lexx1 * (Hx1 - Hx0) + lexx2 * (Hx2 -  Hx0));
        Ry += cell_2.y * (leyy1 * (Hy1 - Hy0) + leyy2 * (Hy2 -  Hy0));
        Rz += cell_2.z * (lezz1 * (Hz1 - Hz0) + lezz2 * (Hz2 -  Hz0));

        // neighbors in Y direction
        idx = j - 1;
        idx = (idx < 0 && wrap.y) ? N.y + idx : idx;
        idx = max(idx, 0);
        linAddr = i * N.y * N.z + idx * N.z + k;

        lexx = lambda_eMul_xx * getMaskUnity(lambda_e_xx, linAddr);
        leyy = lambda_eMul_yy * getMaskUnity(lambda_e_yy, linAddr);
        lezz = lambda_eMul_zz * getMaskUnity(lambda_e_zz, linAddr);

        lexx1 = avgGeomZero(lexx0, lexx);
        leyy1 = avgGeomZero(leyy0, leyy);
        lezz1 = avgGeomZero(lezz0, lezz);

        Hx1 = Hx[linAddr];
        Hy1 = Hy[linAddr];
        Hz1 = Hz[linAddr];

        idx = j + 1;
        idx = (idx == N.y && wrap.y) ? idx - N.y : idx;
        idx = min(idx, N.y - 1);
        linAddr = i * N.y * N.y + idx * N.y + k;

        lexx = lambda_eMul_xx * getMaskUnity(lambda_e_xx, linAddr);
        leyy = lambda_eMul_yy * getMaskUnity(lambda_e_yy, linAddr);
        lezz = lambda_eMul_zz * getMaskUnity(lambda_e_zz, linAddr);

        lexx2 = avgGeomZero(lexx0, lexx);
        leyy2 = avgGeomZero(leyy0, leyy);
        lezz2 = avgGeomZero(lezz0, lezz);

        Hx2 = Hx[linAddr];
        Hy2 = Hy[linAddr];
        Hz2 = Hz[linAddr];

        Rx += cell_2.x * (lexx1 * (Hx1 - Hx0) + lexx2 * (Hx2 -  Hx0));
        Ry += cell_2.y * (leyy1 * (Hy1 - Hy0) + leyy2 * (Hy2 -  Hy0));
        Rz += cell_2.z * (lezz1 * (Hz1 - Hz0) + lezz2 * (Hz2 -  Hz0));

        // Write back to global memory
        tx[I] = -Rx;
        ty[I] = -Ry;
        tz[I] = -Rz;
    }
}

#define BLOCKSIZE 16

__export__  void llbar_nonlocal00nc_async(float* tx, float*  ty, float*  tz,
        float*  hx, float*  hy, float*  hz,

        float* msat0T0,

        float* lambda_e_xx,
        float* lambda_e_yy,
        float* lambda_e_zz,

        const float lambda_eMul_xx,
        const float lambda_eMul_yy,
        const float lambda_eMul_zz,

        const int sx, const int sy, const int sz,
        const float csx, const float csy, const float csz,
        const int pbc_x, const int pbc_y, const int pbc_z,
        CUstream stream)
{

    dim3 gridSize(divUp(sy, BLOCKSIZE), divUp(sz, BLOCKSIZE));
    dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);

    float cellx_2 = (float)(1.0 / ((double)csx * (double)csx));
    float celly_2 = (float)(1.0 / ((double)csy * (double)csy));
    float cellz_2 = (float)(1.0 / ((double)csz * (double)csz));


    float3 cell_2 = make_float3(cellx_2, celly_2, cellz_2);
    int3 N = make_int3(sx, sy, sz);
    int3 wrap = make_int3(pbc_x, pbc_y, pbc_z);

    llbarNonlocal00ncKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (tx, ty, tz,
            hx, hy, hz,

            msat0T0,

            lambda_e_xx,
            lambda_e_yy,
            lambda_e_zz,

            lambda_eMul_xx,
            lambda_eMul_yy,
            lambda_eMul_zz,

            N,
            cell_2,
            wrap);


}

// ========================================

#ifdef __cplusplus
}
#endif
