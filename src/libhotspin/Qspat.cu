#include "Qspat.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "stdio.h"
#include <cuda.h>
#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BLOCKSIZE 16

///@internal
__global__ void QspatKern(float* __restrict__ Q,
                          const float* __restrict__ Ti,
                          const float* __restrict__ lTi, const float* __restrict__ rTi,
                          const float* __restrict__ kMask,
                          const float kMul,
                          const int4 size,
                          const float3 mstep,
                          const int3 pbc,
                          int i)
{

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < size.y && k < size.z)
    {

        int x0 = i * size.w + j * size.z + k;

        int xb1, xf1, x;

        xb1 = (i == 0 && pbc.x == 0) ? i     : i - 1;
        x   = (i == 0 && pbc.x == 0) ? i + 1 : i;
        xf1 = (i == 0 && pbc.x == 0) ? i + 2 : i + 1;

        xb1 = (i == size.x - 1 && pbc.x == 0) ? i - 2 : xb1;
        x   = (i == size.x - 1 && pbc.x == 0) ? i - 1 : x;
        xf1 = (i == size.x - 1 && pbc.x == 0) ? i     : xf1;

        int yb1, yf1, y;
        yb1 = (j == 0 && lTi == NULL) ? j     : j - 1;
        y   = (j == 0 && lTi == NULL) ? j + 1 : j;
        yf1 = (j == 0 && lTi == NULL) ? j + 2 : j + 1;
        yb1 = (j == size.y - 1 && rTi == NULL) ? j - 2 : yb1;
        y   = (j == size.y - 1 && rTi == NULL) ? j - 1 : y;
        yf1 = (j == size.y - 1 && rTi == NULL) ? j     : yf1;

        int zb1, zf1, z;
        zb1 = (k == 0 && pbc.z == 0) ? k     : k - 1;
        z   = (k == 0 && pbc.z == 0) ? k + 1 : k;
        zf1 = (k == 0 && pbc.z == 0) ? k + 2 : k + 1;
        zb1 = (k == size.z - 1 && pbc.z == 0) ? k - 2 : zb1;
        z   = (k == size.z - 1 && pbc.z == 0) ? k - 1 : z;
        zf1 = (k == size.z - 1 && pbc.z == 0) ? k     : zf1;

        xb1 = (xb1 < 0) ?          size.x + xb1 : xb1;
        xf1 = (xf1 > size.x - 1) ? xf1 - size.x : xf1;

        yb1 = (yb1 < 0) ?          size.y + yb1 : yb1;
        yf1 = (yf1 > size.y - 1) ? yf1 - size.y : yf1;

        zb1 = (zb1 < 0) ?          size.z + zb1 : zb1;
        zf1 = (zf1 > size.z - 1) ? zf1 - size.z : zf1;

        int comm = j * size.z + k;

        int3 xn = make_int3(xb1 * size.w + comm,
                            x   * size.w + comm,
                            xf1 * size.w + comm);


        comm = i * size.w + k;

        int3 yn = make_int3(yb1 * size.z + comm,
                            y   * size.z + comm,
                            yf1 * size.z + comm);


        comm = i * size.w + j * size.z;

        int3 zn = make_int3(zb1 + comm,
                            z   + comm,
                            zf1 + comm);


        // Let's use 3-point stencil in the bulk and 3-point forward/backward at the boundary
        float T_b1, T, T_f1;
        float ddT_x, ddT_y, ddT_z;

        float ddT;
        float sum;

        T_b1   = Ti[xn.x];
        T      = Ti[xn.y];
        T_f1   = Ti[xn.z];

        sum    = __fadd_rn(T_b1, T_f1);
        ddT_x = (size.x > 3) ? __fmaf_rn(-2.0f, T, sum) : 0.0f;

        T_b1 = (j > 0 || lTi == NULL) ? Ti[yn.x] : lTi[yn.x];
        T    = Ti[yn.y];
        T_f1 = (j < size.y - 1 || rTi == NULL) ? Ti[yn.z] : rTi[yn.z];

        sum    = __fadd_rn(T_b1, T_f1);
        ddT_y = (size.y > 3) ? __fmaf_rn(-2.0f, T, sum) : 0.0f;

        T_b1 = Ti[zn.x];
        T    = Ti[zn.y];
        T_f1 = Ti[zn.z];

        sum    = __fadd_rn(T_b1, T_f1);
        ddT_z = (size.z > 3) ? __fmaf_rn(-2.0f, T, sum) : 0.0f;

        ddT   = mstep.x * ddT_x + mstep.y * ddT_y + mstep.z * ddT_z;
        // ddT is the laplacian(T)

        float k = (kMask != NULL) ? kMask[x0] * kMul : kMul;

        Q[x0] = k * ddT;
    }
}

__export__ void Qspat_async(float* Q,
                            float* T,
                            float* k,
                            const float kMul,
                            const int sx, const int sy, const int sz,
                            const float csx, const float csy, const float csz,
                            const int pbc_x, const int pbc_y, const int pbc_z,
                            CUstream stream)
{


    dim3 gridSize(divUp(sy, BLOCKSIZE), divUp(sz, BLOCKSIZE));
    dim3 blockSize(BLOCKSIZE, BLOCKSIZE, 1);

    float icsx2 = 1.0f / (csx * csx);
    float icsy2 = 1.0f / (csy * csy);
    float icsz2 = 1.0f / (csz * csz);

    int syz = sy * sz;

    float3 mstep = make_float3(icsx2, icsy2, icsz2);
    int4 size = make_int4(sx, sy, sz, syz);
    int3 pbc = make_int3(pbc_x, pbc_y, pbc_z);

    float* lT = NULL;
    float* rT = NULL;

    for (int i = 0; i < sx; i++)
    {
        QspatKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (Q,
                T,
                lT, rT,
                k,
                kMul,
                size,
                mstep,
                pbc, i);
    }
}

#ifdef __cplusplus
}
#endif
