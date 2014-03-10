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
                              const double lex2Mulmsat0T0Mul_cellSizeX2, const double lex2Mulmsat0T0Mul_cellSizeY2, const double lex2Mulmsat0T0Mul_cellSizeZ2)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i < N0 && j < N1 && k < N2)
    {

        int I = i * N1 * N2 + j * N2 + k;

        double lex02 = getMaskUnity(lexMsk, I) * getMaskUnity(lexMsk, I);
        double lex2, pre1, pre2;

        double3 m0 = make_double3(mx[I], my[I], mz[I]);
        double ms0 = len(m0) * getMaskUnity(msat0T0Msk, I);
        double3 s0 = normalize(m0);

        double Hx, Hy, Hz;
        double ms2, ms1;
        double3 s1, s2;
        double3 m1, m2;

        int linAddr;

        // neighbors in X direction
        int idx = i - 1;
        idx = (idx < 0 && wrap0) ? N0 + idx : idx;
        idx = max(idx, 0);
        linAddr = idx * N1 * N2 + j * N2 + k;
    
        m1 = make_double3(mx[linAddr], my[linAddr], mz[linAddr]);
        ms1 = len(m1) * getMaskUnity(msat0T0Msk, linAddr);
        s1 = normalize(m1);

        lex2 = getMaskUnity(lexMsk, linAddr) * getMaskUnity(lexMsk, linAddr);
        pre1 = avgGeomZero(lex02 * ms0, lex2 * ms1);

        idx = i + 1;
        idx = (idx == N0 && wrap0) ? idx - N0 : idx;
        idx = min(idx, N0 - 1);
        linAddr = idx * N1 * N2 + j * N2 + k;

        m2 = make_double3(mx[linAddr], my[linAddr], mz[linAddr]);
        ms2 = len(m2) * getMaskUnity(msat0T0Msk, linAddr);;
        s2 = normalize(m2);

        lex2 = getMaskUnity(lexMsk, linAddr) * getMaskUnity(lexMsk, linAddr);
        pre2 = avgGeomZero(lex02 * ms0, lex2 * ms2);

        Hx = lex2Mulmsat0T0Mul_cellSizeX2 * (pre1 * (s1.x - s0.x) + pre2 * (s2.x - s0.x));
        Hy = lex2Mulmsat0T0Mul_cellSizeX2 * (pre1 * (s1.y - s0.y) + pre2 * (s2.y - s0.y));
        Hz = lex2Mulmsat0T0Mul_cellSizeX2 * (pre1 * (s1.z - s0.z) + pre2 * (s2.z - s0.z));

        // neighbors in Z direction
        idx = k - 1;
        idx = (idx < 0 && wrap2) ? N2 + idx : idx;
        idx = max(idx, 0);
        linAddr = i * N1 * N2 + j * N2 + idx;
    
        m1 = make_double3(mx[linAddr], my[linAddr], mz[linAddr]);
        ms1 = len(m1) * getMaskUnity(msat0T0Msk, linAddr);
        s1 = normalize(m1);

        lex2 = getMaskUnity(lexMsk, linAddr) * getMaskUnity(lexMsk, linAddr);
        pre1 = avgGeomZero(lex02 * ms0, lex2 * ms1);

        idx = k + 1;
        idx = (idx == N2 && wrap2) ? idx - N2 : idx;
        idx = min(idx, N2 - 1);
        linAddr = i * N1 * N2 + j * N2 + idx;

        m2 = make_double3(mx[linAddr], my[linAddr], mz[linAddr]);
        ms2 = len(m2) * getMaskUnity(msat0T0Msk, linAddr);
        s2 = normalize(m2);

        lex2 = getMaskUnity(lexMsk, linAddr) * getMaskUnity(lexMsk, linAddr);
        pre2 = avgGeomZero(lex02 * ms0, lex2 * ms2);

        Hx += lex2Mulmsat0T0Mul_cellSizeZ2 * (pre1 * (s1.x - s0.x) + pre2 * (s2.x - s0.x));
        Hy += lex2Mulmsat0T0Mul_cellSizeZ2 * (pre1 * (s1.y - s0.y) + pre2 * (s2.y - s0.y));
        Hz += lex2Mulmsat0T0Mul_cellSizeZ2 * (pre1 * (s1.z - s0.z) + pre2 * (s2.z - s0.z));

        // neighbors in Y direction
        idx = j - 1;
        idx = (idx < 0 && wrap1) ? N1 + idx : idx;
        idx = max(idx, 0);
        linAddr = i * N1 * N2 + idx * N2 + k;

        m1 = make_double3(mx[linAddr], my[linAddr], mz[linAddr]);
        ms1 = len(m1) * getMaskUnity(msat0T0Msk, linAddr);
        s1 = normalize(m1);

        lex2 = getMaskUnity(lexMsk, linAddr) * getMaskUnity(lexMsk, linAddr);
        pre1 = avgGeomZero(lex02 * ms0, lex2 * ms1);

        idx = j + 1;
        idx = (idx == N1 && wrap1) ? idx - N1 : idx;
        idx = min(idx, N1 - 1);
        linAddr = i * N1 * N2 + idx * N2 + k;

        m2 = make_double3(mx[linAddr], my[linAddr], mz[linAddr]);
        ms2 = len(m2) * getMaskUnity(msat0T0Msk, linAddr);
        s2 = normalize(m2);

        lex2 = getMaskUnity(lexMsk, linAddr) * getMaskUnity(lexMsk, linAddr);
        pre2 = avgGeomZero(lex02 * ms0, lex2 * ms2);
        
        Hx += lex2Mulmsat0T0Mul_cellSizeY2 * (pre1 * (s1.x - s0.x) + pre2 * (s2.x - s0.x));
        Hy += lex2Mulmsat0T0Mul_cellSizeY2 * (pre1 * (s1.y - s0.y) + pre2 * (s2.y - s0.y));
        Hz += lex2Mulmsat0T0Mul_cellSizeY2 * (pre1 * (s1.z - s0.z) + pre2 * (s2.z - s0.z));

        // Write back to global memory
        hx[I] = Hx;
        hy[I] = Hy;
        hz[I] = Hz;

    }

}


__export__ void exchange6Async(double* hx, double* hy, double* hz, 
                              double* mx, double* my, double* mz, 
                              double* msat0T0, 
                              double* lex, 
                              int N0, int N1Part, int N2, 
                              int periodic0, int periodic1, int periodic2, 
                              double lex2Mulmsat0T0Mul_cellSizeX2, double lex2Mulmsat0T0Mul_cellSizeY2, double lex2Mulmsat0T0Mul_cellSizeZ2, 
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
                                                                            lex2Mulmsat0T0Mul_cellSizeX2, lex2Mulmsat0T0Mul_cellSizeY2, lex2Mulmsat0T0Mul_cellSizeZ2);
}


#ifdef __cplusplus
}
#endif

