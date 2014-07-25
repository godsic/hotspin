#include "exchange6.h"

#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"
#ifdef __cplusplus
extern "C" {
#endif
// full 3D blocks
__global__ void exchange6Kern(double* __restrict__ hx, double* __restrict__  hy, double* __restrict__  hz, 
                              double* __restrict__  mx, double* __restrict__  my, double* __restrict__  mz,
                              double* __restrict__  msat0T0Msk,
                              double* __restrict__  lexMsk,
                              const int N0, const int N1, const int N2,
                              const int wrap0, const int wrap1, const int wrap2,
                              const double msat0T0Mul,
                              const double lex2Mul_cellSizeX2, const double lex2Mul_cellSizeY2, const double lex2Mul_cellSizeZ2)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < N0 && j < N1 && k < N2)
    {

        int I = i * N1 * N2 + j * N2 + k;

        double lex = getMaskUnity(lexMsk, I);
        double lex02 =  lex * lex;
        double lex2, pre1, pre2;

        double ms0 = getMaskUnity(msat0T0Msk, I);
        double3 m0 = make_double3(mx[I] * ms0, my[I] * ms0, mz[I] * ms0);

        double Hx, Hy, Hz;
        double ms2, ms1;
        double3 m1, m2;

        int linAddr;

        // neighbors in X direction
        int idx = i - 1;
        idx = (idx < 0 && wrap0) ? N0 + idx : idx;
        idx = max(idx, 0);
        linAddr = idx * N1 * N2 + j * N2 + k;
    
        ms1 = getMaskUnity(msat0T0Msk, linAddr);
        m1 = make_double3(mx[linAddr] * ms1, my[linAddr] * ms1, mz[linAddr] * ms1);

        lex = getMaskUnity(lexMsk, linAddr);
        lex2 =  lex * lex;
        pre1 = avgGeomZero(lex02, lex2);

        idx = i + 1;
        idx = (idx == N0 && wrap0) ? idx - N0 : idx;
        idx = min(idx, N0 - 1);
        linAddr = idx * N1 * N2 + j * N2 + k;

        ms2 = getMaskUnity(msat0T0Msk, linAddr);
        m2 = make_double3(mx[linAddr] * ms2, my[linAddr] * ms2, mz[linAddr] * ms2);

        lex = getMaskUnity(lexMsk, linAddr);
        lex2 =  lex * lex;
        pre2 = avgGeomZero(lex02, lex2);

        Hx = lex2Mul_cellSizeX2 * (pre1 * (m1.x - m0.x) + pre2 * (m2.x - m0.x));
        Hy = lex2Mul_cellSizeX2 * (pre1 * (m1.y - m0.y) + pre2 * (m2.y - m0.y));
        Hz = lex2Mul_cellSizeX2 * (pre1 * (m1.z - m0.z) + pre2 * (m2.z - m0.z));

        // neighbors in Z direction
        idx = k - 1;
        idx = (idx < 0 && wrap2) ? N2 + idx : idx;
        idx = max(idx, 0);
        linAddr = i * N1 * N2 + j * N2 + idx;

        ms1 = getMaskUnity(msat0T0Msk, linAddr);
        m1 = make_double3(mx[linAddr] * ms1, my[linAddr] * ms1, mz[linAddr] * ms1);

        lex = getMaskUnity(lexMsk, linAddr);
        lex2 =  lex * lex;
        pre1 = avgGeomZero(lex02, lex2);

        idx = k + 1;
        idx = (idx == N2 && wrap2) ? idx - N2 : idx;
        idx = min(idx, N2 - 1);
        linAddr = i * N1 * N2 + j * N2 + idx;

        ms2 = getMaskUnity(msat0T0Msk, linAddr);
        m2 = make_double3(mx[linAddr] * ms2, my[linAddr] * ms2, mz[linAddr] * ms2);

        lex = getMaskUnity(lexMsk, linAddr);
        lex2 =  lex * lex;
        pre2 = avgGeomZero(lex02, lex2);

        Hx += lex2Mul_cellSizeZ2 * (pre1 * (m1.x - m0.x) + pre2 * (m2.x - m0.x));
        Hy += lex2Mul_cellSizeZ2 * (pre1 * (m1.y - m0.y) + pre2 * (m2.y - m0.y));
        Hz += lex2Mul_cellSizeZ2 * (pre1 * (m1.z - m0.z) + pre2 * (m2.z - m0.z));

        // neighbors in Y direction
        idx = j - 1;
        idx = (idx < 0 && wrap1) ? N1 + idx : idx;
        idx = max(idx, 0);
        linAddr = i * N1 * N2 + idx * N2 + k;

        ms1 = getMaskUnity(msat0T0Msk, linAddr);
        m1 = make_double3(mx[linAddr] * ms1, my[linAddr] * ms1, mz[linAddr] * ms1);

        lex = getMaskUnity(lexMsk, linAddr);
        lex2 =  lex * lex;
        pre1 = avgGeomZero(lex02, lex2);

        idx = j + 1;
        idx = (idx == N1 && wrap1) ? idx - N1 : idx;
        idx = min(idx, N1 - 1);
        linAddr = i * N1 * N2 + idx * N2 + k;

        ms2 = getMaskUnity(msat0T0Msk, linAddr);
        m2 = make_double3(mx[linAddr] * ms2, my[linAddr] * ms2, mz[linAddr] * ms2);

        lex = getMaskUnity(lexMsk, linAddr);
        lex2 =  lex * lex;
        pre2 = avgGeomZero(lex02, lex2);
        
        Hx += lex2Mul_cellSizeY2 * (pre1 * (m1.x - m0.x) + pre2 * (m2.x - m0.x));
        Hy += lex2Mul_cellSizeY2 * (pre1 * (m1.y - m0.y) + pre2 * (m2.y - m0.y));
        Hz += lex2Mul_cellSizeY2 * (pre1 * (m1.z - m0.z) + pre2 * (m2.z - m0.z));

        // Write back to global memory
        hx[I] = msat0T0Mul * Hx;
        hy[I] = msat0T0Mul * Hy;
        hz[I] = msat0T0Mul * Hz;

    }

}


__export__ void exchange6Async(double* hx, double* hy, double* hz, 
                              double* mx, double* my, double* mz, 
                              double* msat0T0, 
                              double* lex, 
                              int N0, int N1Part, int N2, 
                              int periodic0, int periodic1, int periodic2,
                              double msat0T0Mul,
                              double lex2Mul_cellSizeX2, double lex2Mul_cellSizeY2, double lex2Mul_cellSizeZ2, 
                              CUstream streams)
{
    dim3 gridsize, blocksize;

    make3dconf(N0, N1Part, N2, &gridsize, &blocksize);

    exchange6Kern <<< gridsize, blocksize, 0, cudaStream_t(streams)>>>(hx, hy, hz,
                                                                            mx, my, mz, 
                                                                            msat0T0, 
                                                                            lex, 
                                                                            N0, N1Part, N2, 
                                                                            periodic0, periodic1, periodic2, 
                                                                            msat0T0Mul,
                                                                            lex2Mul_cellSizeX2, lex2Mul_cellSizeY2, lex2Mul_cellSizeZ2);
}


#ifdef __cplusplus
}
#endif

