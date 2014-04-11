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
__global__ void tensSYMMVecMulKern(double* __restrict__ dstX, double* __restrict__ dstY, double* __restrict__ dstZ,
							   double* __restrict__ srcX, double* __restrict__ srcY, double* __restrict__ srcZ,
							   double* __restrict__ kernXX, double* __restrict__ kernYY, double* __restrict__ kernZZ,
							   double* __restrict__ kernYZ, double* __restrict__ kernXZ, double* __restrict__ kernXY,
							   const double srcMul, 
							   const int3 N)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < N.x && j < N.y && k < N.z)
    {
    	int I = i * N.y * N.z + j * N.z + k;
    	int e = 2 * I; 

        double RMX = srcMul * srcX[e + 0];
        double IMX = srcMul * srcX[e + 1];

        double RMY = srcMul * srcY[e + 0];
        double IMY = srcMul * srcY[e + 1];

        double RMZ = srcMul * srcZ[e + 0];
        double IMZ = srcMul * srcZ[e + 1];

        double signXY = 1.0;
        double signXZ = 1.0;
        double signYZ = 1.0;

        signXY = (j > N.y / 2) ? -signXY : signXY;
        signYZ = (j > N.y / 2) ? -signYZ : signYZ;
		int jj = (j > N.y / 2) ? N.y - j : j;
        
        signXY = (i > N.x / 2) ? -signXY : signXY;
        signXZ = (i > N.x / 2) ? -signXZ : signXZ;
        int ii = (i > N.x / 2) ? N.x - i : i;

        I = ii * (N.y / 2 + 1) * N.z + jj * N.z + k;

        double KXX = getMaskZero(kernXX, I);
        double KXZ = signXZ * getMaskZero(kernXZ, I);
        double KXY = signXY * getMaskZero(kernXY, I);

        dstX[e + 0] = KXX * RMX + KXY * RMY + KXZ * RMZ;
        dstX[e + 1] = KXX * IMX + KXY * IMY + KXZ * IMZ;

        double KYZ = signYZ * getMaskZero(kernYZ, I);
        double KYY = getMaskZero(kernYY, I);

        dstY[e + 0] = KXY * RMX + KYY * RMY + KYZ * RMZ;
        dstY[e + 1] = KXY * IMX + KYY * IMY + KYZ * IMZ;

        double KZZ = getMaskZero(kernZZ, I);

        dstZ[e + 0] = KXZ * RMX + KYZ * RMY + KZZ * RMZ;
        dstZ[e + 1] = KXZ * IMX + KYZ * IMY + KZZ * IMZ;

    }
}

__export__ void tensSYMMVecMul(double* dstX, double* dstY, double* dstZ,
						 double* srcX, double* srcY, double* srcZ,
						 double* kernXX, double* kernYY, double* kernZZ,
						 double* kernYZ, double* kernXZ, double* kernXY,
						 const double srcMul,
						 const int Nx, const int Ny, const int Nz,
						 CUstream stream)
{
    dim3 gridsize, blocksize;

    make3dconf(Nx, Ny, Nz, &gridsize, &blocksize);

    const int3 N = make_int3(Nx, Ny, Nz);

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

