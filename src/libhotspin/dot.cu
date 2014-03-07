#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"
#ifdef __cplusplus
extern "C" {
#endif

///@internal
__global__ void dotMaskKern(float* __restrict__ dst,
                            float* __restrict__ ax, float* __restrict__ ay, float* __restrict__ az,
                            float* __restrict__ bx, float* __restrict__ by, float* __restrict__ bz,
                            float axMul, float ayMul, float azMul,
                            float bxMul, float byMul, float bzMul,
                            int Npart)
{
    int i = threadindex;
    if (i < Npart)
    {
        float3 a = make_float3(axMul * getMaskUnity(ax, i), ayMul * getMaskUnity(ay, i), azMul * getMaskUnity(az, i));
        float3 b = make_float3(bxMul * getMaskUnity(bx, i), byMul * getMaskUnity(by, i), bzMul * getMaskUnity(bz, i));
        dst[i] = dotf(a, b);
    }
}

///@internal
__global__ void dotKern(float* __restrict__ dst,
                        float* __restrict__ ax, float* __restrict__ ay, float* __restrict__ az,
                        float* __restrict__ bx, float* __restrict__ by, float* __restrict__ bz,
                        int Npart)
{
    int i = threadindex;
    if (i < Npart)
    {
        float3 a = make_float3(ax[i], ay[i], az[i]);
        float3 b = make_float3(bx[i], by[i], bz[i]);
        dst[i] = dotf(a, b);
    }
}



///@internal
// if b || c, then dst < 0.
__global__ void dotSignKern(float* __restrict__ dst,
                            float* __restrict__ ax, float* __restrict__ ay, float* __restrict__ az,
                            float* __restrict__ bx, float* __restrict__ by, float* __restrict__ bz,
                            float* __restrict__ cx, float* __restrict__ cy, float* __restrict__ cz,
                            int Npart)
{
    int i = threadindex;
    if (i < Npart)
    {
        float3 a = make_float3(ax[i], ay[i], az[i]);
        float3 b = make_float3(bx[i], by[i], bz[i]);
        float3 c = make_float3(cx[i], cy[i], cz[i]);
        float dotP = dotf(a, b);
        float sign = -dotf(b, c); // !!!
        dst[i] = copysign(dotP, sign);
    }
}


__export__ void dotMaskAsync(float* dst, 
                             float* ax, float* ay, float* az, 
                             float* bx, float* by, float* bz, 
                             float axMul, float ayMul, float azMul,
                             float bxMul, float byMul, float bzMul,
                             CUstream stream, int Npart)
{
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    dotMaskKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (dst, 
                                                                      ax, ay, az, 
                                                                      bx, by, bz, 
                                                                      axMul, ayMul, azMul,
                                                                      bxMul, byMul, bzMul,
                                                                      Npart);
}


__export__ void dotAsync(float* dst, float* ax, float* ay, float* az, float* bx, float* by, float* bz, CUstream stream, int Npart)
{
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    dotKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (dst, ax, ay, az, bx, by, bz, Npart);
}

__export__ void dotSignAsync(float* dst, float* ax, float* ay, float* az, float* bx, float* by, float* bz, float* cx, float* cy, float* cz, CUstream stream, int Npart)
{
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    dotSignKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (dst,
            ax, ay, az,
            bx, by, bz,
            cx, cy, cz,
            Npart);
}


#ifdef __cplusplus
}
#endif
