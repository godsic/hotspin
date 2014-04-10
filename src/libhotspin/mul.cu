#include "mul.h"

#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal
__global__ void mulKern(double* dst, double* a, double* b, int Npart)
{
    int i = threadindex;
    if (i < Npart)
    {
        dst[i] = a[i] * b[i];
    }
}


__export__ void mulAsync(double* dst, double* a, double* b, CUstream stream, int Npart)
{
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    mulKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (dst, a, b, Npart);
}

///@internal
__global__ void tensSYMMVecMulKern(double* dstX, double* dstY, double* dstZ,
							   double* srcX, double* srcY, double* srcZ,
							   double* kernXX, double* kernYY, double* kernZZ,
							   double* kernYZ, double* kernXZ, double* kernXY,
							   double3 srcMul, 
							   int3 N)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < N.x && j < N.y && k < N.z)
    {
    	int I = i * N.y * N.z + j * N.z + k;
    	int e = 2 * I; 

        double KXX = getMaskZero(kernXX, I);
        double KYY = getMaskZero(kernYY, I);
        double KZZ = getMaskZero(kernZZ, I);

        double KYZ = getMaskZero(kernYZ, I);
        double KZY = KYZ;

        double KXZ = getMaskZero(kernXZ, I);
        double KZX = KXZ;

        double KXY = getMaskZero(kernXY, I);
        double KYX = KXY;

        double RMX = srcX[e + 0];
        double IMX = srcX[e + 1];

        double RMY = srcY[e + 0];
        double IMY = srcY[e + 1];

        double RMZ = srcZ[e + 0];
        double IMZ = srcZ[e + 1];

        dstX[e + 0] = KXX * srcMul.x * RMX + KXY * srcMul.y * RMY + KXZ * srcMul.z * RMZ;
        dstX[e + 1] = KXX * srcMul.x * IMX + KXY * srcMul.y * IMY + KXZ * srcMul.z * IMZ;

        dstY[e + 0] = KYX * srcMul.x * RMX + KYY * srcMul.y * RMY + KYZ * srcMul.z * RMZ;
        dstY[e + 1] = KYX * srcMul.x * IMX + KYY * srcMul.y * IMY + KYZ * srcMul.z * IMZ;

        dstZ[e + 0] = KZX * srcMul.x * RMX + KZY * srcMul.y * RMY + KZZ * srcMul.z * RMZ;
        dstZ[e + 1] = KZX * srcMul.x * IMX + KZY * srcMul.y * IMY + KZZ * srcMul.z * IMZ;

    }
}

__export__ void tensSYMMVecMul(double* dstX, double* dstY, double* dstZ,
						 double* srcX, double* srcY, double* srcZ,
						 double* kernXX, double* kernYY, double* kernZZ,
						 double* kernYZ, double* kernXZ, double* kernXY,
						 double srcMulX, double srcMulY, double srcMulZ,
						 int Nx, int Ny, int Nz,
						 CUstream stream)
{
    dim3 gridsize, blocksize;

    make3dconf(Nx, Ny, Nz, &gridsize, &blocksize);

    int3 N = make_int3(Nx, Ny, Nz);
    double3 srcMul = make_double3(srcMulX, srcMulY, srcMulZ);

    tensSYMMVecMulKern <<< gridsize, blocksize, 0, cudaStream_t(stream)>>> (dstX, dstY, dstZ,
    																	   srcX, srcY, srcZ,
    																	   kernXX, kernYY, kernZZ,
    																	   kernYZ, kernXZ, kernXY,
    																	   srcMul,
    																	   N);
}


#ifdef __cplusplus
}
#endif

