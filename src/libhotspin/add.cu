
#include "add.h"

#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal
__global__ void addKern(double* dst, double* a, double* b, int Npart)
{
    int i = threadindex;
    if (i < Npart)
    {
        dst[i] = a[i] + b[i];
    }
}


__export__ void addAsync(double* dst, double* a, double* b, CUstream stream, int Npart)
{
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    addKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (dst, a, b, Npart);
}

///@internal
__global__ void addmaddKern(double* dst, double* a, double* b, double* c, double mul, int Npart)
{
    int i = threadindex;
    if (i < Npart)
    {
        dst[i] = a[i] + mul * (b[i] + c[i]);
    }
}

///@internal
__global__ void maddKern(double* dst, double* a, double* b, double mulB, int Npart)
{
    int i = threadindex;
    if (i < Npart)
    {
        dst[i] = a[i] + mulB * b[i];
    }
}

__global__ void maddScalarKern(double* dst, double* a, double mulB, int Npart)
{
    int i = threadindex;
    if (i < Npart)
    {
        dst[i] = a[i] + mulB;
    }
}

///@internal

__global__ void vecMaddKern(double*  dstx, double*  dsty, double*  dstz,
                            double*  ax, double*  ay, double*  az,
                            double*  bx, double*  by, double*  bz,
                            double mulBx, double mulBy, double mulBz,
                            int Npart)
{
    int i = threadindex;

    if (i < Npart)
    {
        dstx[i] = ax[i] + mulBx * bx[i];
        dsty[i] = ay[i] + mulBy * by[i];
        dstz[i] = az[i] + mulBz * bz[i];
    }
}

__global__ void vecMaddScalarKern(double*  dstx, double*  dsty, double*  dstz,
                                  double*  ax, double*  ay, double*  az,
                                  double mulBx, double mulBy, double mulBz,
                                  int Npart)
{
    int i = threadindex;

    if (i < Npart)
    {
        dstx[i] = ax[i] + mulBx;
        dsty[i] = ay[i] + mulBy;
        dstz[i] = az[i] + mulBz;
    }
}

__export__ void addMaddAsync(double* dst, double* a, double* b, double* c, double mul, CUstream stream, int NPart)
{
    dim3 gridSize, blockSize;
    make1dconf(NPart, &gridSize, &blockSize);
    addmaddKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (dst, a, b, c, mul, NPart);
}
__export__ void maddAsync(double* dst, double* a, double* b, double mulB, CUstream stream, int Npart)
{
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    if (b == NULL)
    {
        maddScalarKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (dst, a, mulB, Npart);
    }
    else
    {
        maddKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (dst, a, b, mulB, Npart);
    }
}
__export__ void vecMaddAsync(double* dstx, double* dsty, double* dstz,
                             double* ax, double* ay, double* az,
                             double* bx, double* by, double* bz,
                             double mulBx, double mulBy, double mulBz,
                             CUstream stream, int Npart)
{
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    if (bx == NULL)
    {
        vecMaddScalarKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (dstx, dsty, dstz,
                ax, ay, az,
                mulBx,   mulBy,   mulBz,
                Npart);
    }
    else
    {
        vecMaddKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (dstx, dsty, dstz,
                ax, ay, az,
                bx, by, bz,
                mulBx,   mulBy,   mulBz,
                Npart);
    }
}

///@internal
__global__ void madd1Kern(double* a, double* b, double mulB, int Npart)
{
    int i = threadindex;
    if (i < Npart)
    {
        double B = (b == NULL) ? 1.0 : b[i];
        a[i] += mulB * B;
    }
}


__export__ void madd1Async(double* a, double* b, double mulB, CUstream stream, int Npart)
{
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    madd1Kern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (a, b, mulB, Npart);
}

///@internal
__global__ void madd2Kern(double* a, double* b, double mulB, double* c, double mulC, int Npart)
{
    int i = threadindex;
    if (i < Npart)
    {
        double B = (b == NULL) ? 1.0 : b[i];
        double C = (c == NULL) ? 1.0 : c[i];
        a[i] += mulB * B + mulC * C;
    }
}


__export__ void madd2Async(double* a, double* b, double mulB, double* c, double mulC, CUstream stream, int Npart)
{
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    madd2Kern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (a, b, mulB, c, mulC, Npart);
}


__global__ void cmaddKern(double* dst, double a, double b, double* kern, double* src, int NComplexPart)
{

    int i = threadindex; // complex index
    int e = 2 * i; // real index

    if(i < NComplexPart)
    {

        double Sa = src[e  ];
        double Sb = src[e + 1];

        double k = kern[i];

        dst[e  ] += k * (a * Sa - b * Sb);
        dst[e + 1] += k * (b * Sa + a * Sb);
    }

    return;
}

__export__ void cmaddAsync(double* dst, double a, double b, double* kern, double* src, CUstream stream, int NComplexPart)
{
    dim3 gridSize, blockSize;
    make1dconf(NComplexPart, &gridSize, &blockSize);
    cmaddKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (dst, a, b, kern, src, NComplexPart);
}

///@internal
__global__ void linearCombination2Kern(double* dst, double* a, double mulA, double* b, double mulB, int NPart)
{
    int i = threadindex;
    if (i < NPart)
    {
        dst[i] = mulA * a[i] + mulB * b[i];
    }
}

__export__ void linearCombination2Async(double* dst, double* a, double mulA, double* b, double mulB, CUstream stream, int NPart)
{
    dim3 gridSize, blockSize;
    make1dconf(NPart, &gridSize, &blockSize);
    linearCombination2Kern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (dst, a, mulA, b, mulB, NPart);
}



///@internal
__global__ void linearCombination3Kern(double* dst, double* a, double mulA, double* b, double mulB, double* c, double mulC, int NPart)
{
    int i = threadindex;
    if (i < NPart)
    {
        dst[i] = mulA * a[i] + mulB * b[i] + mulC * c[i];
    }
}

__export__ void linearCombination3Async(double* dst, double* a, double mulA, double* b, double mulB, double* c, double mulC, CUstream stream, int NPart)
{
    dim3 gridSize, blockSize;
    make1dconf(NPart, &gridSize, &blockSize);
    linearCombination3Kern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (dst, a, mulA, b, mulB, c, mulC, NPart);
}

#ifdef __cplusplus
}
#endif
