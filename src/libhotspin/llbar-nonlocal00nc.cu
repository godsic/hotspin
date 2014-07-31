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

        if (msat0T0 == 0.0)
        {
            tx[I] = 0.0;
            ty[I] = 0.0;
            tz[I] = 0.0;
            return;
        }

        double lexx0 = getMaskUnity(lambda_e_xx, I);
        double leyy0 = getMaskUnity(lambda_e_yy, I);
        double lezz0 = getMaskUnity(lambda_e_zz, I);

        double lexx, leyy, lezz;
        double lexx1, leyy1, lezz1;
        double lexx2, leyy2, lezz2;

        double Hx0 = __mu0 * Hx[I]; 
        double Hx1, Hx2;

        double Hy0 = __mu0 * Hy[I];
        double Hy1, Hy2;

        double Hz0 = __mu0 * Hz[I];
        double Hz1, Hz2;

        double Rx, Ry, Rz;

        double prex, prey, prez;

        int linAddr;

        // neighbors in X direction
        int idx = i - 1;
        idx = (idx < 0 && wrap.x) ? N.x + idx : idx;
        idx = max(idx, 0);
        linAddr = idx * N.y * N.z + j * N.z + k;

        prex = lambda_eMul_xx * cell_2.x;
        prey = lambda_eMul_yy * cell_2.x;
        prez = lambda_eMul_zz * cell_2.x;

        lexx = getMaskUnity(lambda_e_xx, linAddr);
        leyy = getMaskUnity(lambda_e_yy, linAddr);
        lezz = getMaskUnity(lambda_e_zz, linAddr);

        lexx1 = avgGeomZero(lexx0, lexx);
        leyy1 = avgGeomZero(leyy0, leyy);
        lezz1 = avgGeomZero(lezz0, lezz);

        Hx1 = __mu0 * Hx[linAddr];
        Hy1 = __mu0 * Hy[linAddr];
        Hz1 = __mu0 * Hz[linAddr];

        idx = i + 1;
        idx = (idx == N.x && wrap.x) ? idx - N.x : idx;
        idx = min(idx, N.x - 1);
        linAddr = idx * N.y * N.z + j * N.z + k;

        lexx = getMaskUnity(lambda_e_xx, linAddr);
        leyy = getMaskUnity(lambda_e_yy, linAddr);
        lezz = getMaskUnity(lambda_e_zz, linAddr);

        lexx2 = avgGeomZero(lexx0, lexx);
        leyy2 = avgGeomZero(leyy0, leyy);
        lezz2 = avgGeomZero(lezz0, lezz);

        Hx2 = __mu0 * Hx[linAddr];
        Hy2 = __mu0 * Hy[linAddr];
        Hz2 = __mu0 * Hz[linAddr];
        
        Rx = prex * ((lexx1 * Hx1 + lexx2 * Hx2) - Hx0 * (lexx1 + lexx2));
        Ry = prey * ((leyy1 * Hy1 + leyy2 * Hy2) - Hy0 * (leyy1 + leyy2));
        Rz = prez * ((lezz1 * Hz1 + lezz2 * Hz2) - Hz0 * (lezz1 + lezz2));

        // neighbors in Z direction
        idx = k - 1;
        idx = (idx < 0 && wrap.z) ? N.z + idx : idx;
        idx = max(idx, 0);
        linAddr = i * N.y * N.z + j * N.z + idx;

        prex = lambda_eMul_xx * cell_2.z;
        prey = lambda_eMul_yy * cell_2.z;
        prez = lambda_eMul_zz * cell_2.z;

        lexx = getMaskUnity(lambda_e_xx, linAddr);
        leyy = getMaskUnity(lambda_e_yy, linAddr);
        lezz = getMaskUnity(lambda_e_zz, linAddr);

        lexx1 = avgGeomZero(lexx0, lexx);
        leyy1 = avgGeomZero(leyy0, leyy);
        lezz1 = avgGeomZero(lezz0, lezz);

        Hx1 = __mu0 * Hx[linAddr];
        Hy1 = __mu0 * Hy[linAddr];
        Hz1 = __mu0 * Hz[linAddr];

        idx = k + 1;
        idx = (idx == N.z && wrap.z) ? idx - N.z : idx;
        idx = min(idx, N.z - 1);
        linAddr = i * N.y * N.z + j * N.z + idx;

        lexx = getMaskUnity(lambda_e_xx, linAddr);
        leyy = getMaskUnity(lambda_e_yy, linAddr);
        lezz = getMaskUnity(lambda_e_zz, linAddr);

        lexx2 = avgGeomZero(lexx0, lexx);
        leyy2 = avgGeomZero(leyy0, leyy);
        lezz2 = avgGeomZero(lezz0, lezz);

        Hx2 = __mu0 * Hx[linAddr];
        Hy2 = __mu0 * Hy[linAddr];
        Hz2 = __mu0 * Hz[linAddr];

        Rx += prex * ((lexx1 * Hx1 + lexx2 * Hx2) - Hx0 * (lexx1 + lexx2));
        Ry += prey * ((leyy1 * Hy1 + leyy2 * Hy2) - Hy0 * (leyy1 + leyy2));
        Rz += prez * ((lezz1 * Hz1 + lezz2 * Hz2) - Hz0 * (lezz1 + lezz2));

        // neighbors in Y direction
        idx = j - 1;
        idx = (idx < 0 && wrap.y) ? N.y + idx : idx;
        idx = max(idx, 0);
        linAddr = i * N.y * N.z + idx * N.z + k;

        prex = lambda_eMul_xx * cell_2.y;
        prey = lambda_eMul_yy * cell_2.y;
        prez = lambda_eMul_zz * cell_2.y;

        lexx = getMaskUnity(lambda_e_xx, linAddr);
        leyy = getMaskUnity(lambda_e_yy, linAddr);
        lezz = getMaskUnity(lambda_e_zz, linAddr);

        lexx1 = avgGeomZero(lexx0, lexx);
        leyy1 = avgGeomZero(leyy0, leyy);
        lezz1 = avgGeomZero(lezz0, lezz);

        Hx1 = __mu0 * Hx[linAddr];
        Hy1 = __mu0 * Hy[linAddr];
        Hz1 = __mu0 * Hz[linAddr];

        idx = j + 1;
        idx = (idx == N.y && wrap.y) ? idx - N.y : idx;
        idx = min(idx, N.y - 1);
        linAddr = i * N.y * N.y + idx * N.y + k;

        lexx = getMaskUnity(lambda_e_xx, linAddr);
        leyy = getMaskUnity(lambda_e_yy, linAddr);
        lezz = getMaskUnity(lambda_e_zz, linAddr);

        lexx2 = avgGeomZero(lexx0, lexx);
        leyy2 = avgGeomZero(leyy0, leyy);
        lezz2 = avgGeomZero(lezz0, lezz);

        Hx2 = __mu0 * Hx[linAddr];
        Hy2 = __mu0 * Hy[linAddr];
        Hz2 = __mu0 * Hz[linAddr];

        Rx += prex * ((lexx1 * Hx1 + lexx2 * Hx2) - Hx0 * (lexx1 + lexx2));
        Ry += prey * ((leyy1 * Hy1 + leyy2 * Hy2) - Hy0 * (leyy1 + leyy2));
        Rz += prez * ((lezz1 * Hz1 + lezz2 * Hz2) - Hz0 * (lezz1 + lezz2));

        // Write back to global memory

        prex = 1.0 / __mu0;

        tx[I] = - prex * Rx;
        ty[I] = - prex * Ry;
        tz[I] = - prex * Rz;
    }
}

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
    dim3 gridsize, blocksize;
    make3dconf(sx, sy, sz, &gridsize, &blocksize);

    double cellx_2 = (double)(1.0 / ((double)csx * (double)csx));
    double celly_2 = (double)(1.0 / ((double)csy * (double)csy));
    double cellz_2 = (double)(1.0 / ((double)csz * (double)csz));


    double3 cell_2 = make_double3(cellx_2, celly_2, cellz_2);
    int3 N = make_int3(sx, sy, sz);
    int3 wrap = make_int3(pbc_x, pbc_y, pbc_z);

    llbarNonlocal00ncKern <<< gridsize, blocksize, 0, cudaStream_t(stream)>>> (tx, ty, tz,
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

#ifdef __cplusplus
}
#endif
