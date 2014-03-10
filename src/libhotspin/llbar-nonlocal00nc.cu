#include "llbar-nonlocal00nc.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "stdio.h"
#include <cuda.h>
#include "common_func.h"
#ifdef __cplusplus
extern "C" {
#endif

__global__ void llbarNonlocal00ncKern(double* __restrict__ tx, double* __restrict__ ty, double* __restrict__ tz,

                                      double* __restrict__ Hx, double* __restrict__ Hy, double* __restrict__ Hz,

                                      double* __restrict__ msat0T0Msk,

                                      double* __restrict__ lambda_e_xx,
                                      double* __restrict__ lambda_e_yy,
                                      double* __restrict__ lambda_e_zz,

                                      const double lambda_eMul_xx,
                                      const double lambda_eMul_yy,
                                      const double lambda_eMul_zz,

                                      const int3 N,
                                      const double3 cell_2,
                                      const int3 wrap)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < N.x && j < N.y && k < N.z)
    {

        int I = i * N.y * N.z + j * N.z + k;

        double msat0T0 = getMaskUnity(msat0T0Msk, I);
        // make sure there is no damping in vacuum!
        if (msat0T0 == 0.0)
        {
            tx[I] = 0.0;
            ty[I] = 0.0;
            tz[I] = 0.0;
            return;
        }

        // Second-order derivative 3-points stencil
//==================================================================================================

        double lexx0 = lambda_eMul_xx * getMaskUnity(lambda_e_xx, I);
        double leyy0 = lambda_eMul_yy * getMaskUnity(lambda_e_yy, I);
        double lezz0 = lambda_eMul_zz * getMaskUnity(lambda_e_zz, I);

        double lexx, leyy, lezz;
        double lexx1, leyy1, lezz1;
        double lexx2, leyy2, lezz2;

        double Hx0 = Hx[I]; // mag component of central cell
        double Hx1, Hx2;

        double Hy0 = Hy[I]; // mag component of central cell
        double Hy1, Hy2;

        double Hz0 = Hz[I]; // mag component of central cell
        double Hz1, Hz2;

        double Rx, Ry, Rz;

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

__export__  void llbar_nonlocal00nc_async(double* tx, double*  ty, double*  tz,
        double*  hx, double*  hy, double*  hz,

        double* msat0T0,

        double* lambda_e_xx,
        double* lambda_e_yy,
        double* lambda_e_zz,

        const double lambda_eMul_xx,
        const double lambda_eMul_yy,
        const double lambda_eMul_zz,

        const int sx, const int sy, const int sz,
        const double csx, const double csy, const double csz,
        const int pbc_x, const int pbc_y, const int pbc_z,
        CUstream stream)
{

    dim3 gridSize(divUp(sy, BLOCKSIZE), divUp(sz, BLOCKSIZE));
    dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);

    double cellx_2 = (double)(1.0 / ((double)csx * (double)csx));
    double celly_2 = (double)(1.0 / ((double)csy * (double)csy));
    double cellz_2 = (double)(1.0 / ((double)csz * (double)csz));


    double3 cell_2 = make_double3(cellx_2, celly_2, cellz_2);
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
