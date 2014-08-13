#include "Qspat.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "stdio.h"
#include <cuda.h>
#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal
__global__ void QspatKern(double* __restrict__ Q,
                          double* __restrict__ T,
                          double* __restrict__ kMask,
                          const double kMul,
                          const int3 size,
                          const double3 cell_2,
                          const int3 pbc)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < size.x && j < size.y && k < size.z)
    {   
        int I = i * size.y * size.z + j * size.z + k;

        double ddT;
        double T0, T1, T2;
        double k0, k1, k2;
        double pre1, pre2;
        int idx, linAddr;

        T0 = T[I];
        k0 = getMaskUnity(kMask, I);

        // neighbors in X direction
        idx = i - 1;
        idx = (idx < 0 && pbc.x) ? size.x + idx : idx;
        idx = max(idx, 0);
        linAddr = idx * size.y * size.z + j * size.z + k;

        k1 = getMaskUnity(kMask, linAddr);
        T1 = T[linAddr];
        pre1 = avgGeomZero(k0, k1);

        idx = i + 1;
        idx = (idx == size.x && pbc.x) ? idx - size.x : idx;
        idx = min(idx, size.x - 1);
        linAddr = idx * size.y * size.z + j * size.z + k;

        k2 = getMaskUnity(kMask, linAddr);
        T2 = T[linAddr];
        pre2 = avgGeomZero(k0, k2);
        
        ddT = cell_2.x * ((pre1 * T1 + pre2 * T2) - (pre1 * T0 + pre2 * T0));

        // neighbors in Z direction
        idx = k - 1;
        idx = (idx < 0 && pbc.z) ? size.z + idx : idx;
        idx = max(idx, 0);
        linAddr = i * size.y * size.z + j * size.z + idx;

        k1 = getMaskUnity(kMask, linAddr);
        T1 = T[linAddr];
        pre1 = avgGeomZero(k0, k1);

        idx = k + 1;
        idx = (idx == size.z && pbc.z) ? idx - size.z : idx;
        idx = min(idx, size.z - 1);
        linAddr = i * size.y * size.z + j * size.z + idx;

        k2 = getMaskUnity(kMask, linAddr);
        T2 = T[linAddr];
        pre2 = avgGeomZero(k0, k2);

        ddT += cell_2.z * ((pre1 * T1 + pre2 * T2) - (pre1 * T0 + pre2 * T0));

        // neighbors in Y direction
        idx = j - 1;
        idx = (idx < 0 && pbc.y) ? size.y + idx : idx;
        idx = max(idx, 0);
        linAddr = i * size.y * size.z + idx * size.z + k;

        k1 = getMaskUnity(kMask, linAddr);
        T1 = T[linAddr];
        pre1 = avgGeomZero(k0, k1);

        idx = j + 1;
        idx = (idx == size.y && pbc.y) ? idx - size.y : idx;
        idx = min(idx, size.y - 1);
        linAddr = i * size.y * size.z + idx * size.z + k;

        k2 = getMaskUnity(kMask, linAddr);
        T2 = T[linAddr];
        pre2 = avgGeomZero(k0, k2);

        ddT += cell_2.y * ((pre1 * T1 + pre2 * T2) - (pre1 * T0 + pre2 * T0));

        Q[I] = kMul * ddT;
    }
}

__export__ void Qspat_async(double* Q,
                            double* T,
                            double* k,
                            const double kMul,
                            const int sx, const int sy, const int sz,
                            const double csx, const double csy, const double csz,
                            const int pbc_x, const int pbc_y, const int pbc_z,
                            CUstream stream)
{


    dim3 gridsize, blocksize;
    make3dconf(sx, sy, sz, &gridsize, &blocksize);

    double cellx_2 = 1.0 / (csx * csx);
    double celly_2 = 1.0 / (csy * csy);
    double cellz_2 = 1.0 / (csz * csz);

    double3 cell_2 = make_double3(cellx_2, celly_2, cellz_2);
    int3 size = make_int3(sx, sy, sz);
    int3 pbc = make_int3(pbc_x, pbc_y, pbc_z);

    QspatKern <<< gridsize, blocksize, 0, cudaStream_t(stream)>>> (Q,
               T,
               k,
               kMul,
               size,
               cell_2,
               pbc);
}

#ifdef __cplusplus
}
#endif
