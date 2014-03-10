#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"
#ifdef __cplusplus
extern "C" {
#endif

///@internal
__global__ void dotMaskKern(double* __restrict__ dst,
                            double* __restrict__ ax, double* __restrict__ ay, double* __restrict__ az,
                            double* __restrict__ bx, double* __restrict__ by, double* __restrict__ bz,
                            double axMul, double ayMul, double azMul,
                            double bxMul, double byMul, double bzMul,
                            int Npart)
{
    int i = threadindex;
    if (i < Npart)
    {
        double3 a = make_double3(axMul * getMaskUnity(ax, i), ayMul * getMaskUnity(ay, i), azMul * getMaskUnity(az, i));
        double3 b = make_double3(bxMul * getMaskUnity(bx, i), byMul * getMaskUnity(by, i), bzMul * getMaskUnity(bz, i));
        dst[i] = dot(a, b);
    }
}

///@internal
__global__ void dotKern(double* __restrict__ dst,
                        double* __restrict__ ax, double* __restrict__ ay, double* __restrict__ az,
                        double* __restrict__ bx, double* __restrict__ by, double* __restrict__ bz,
                        int Npart)
{
    int i = threadindex;
    if (i < Npart)
    {
        double3 a = make_double3(ax[i], ay[i], az[i]);
        double3 b = make_double3(bx[i], by[i], bz[i]);
        dst[i] = dot(a, b);
    }
}



///@internal
// if b || c, then dst < 0.
__global__ void dotSignKern(double* __restrict__ dst,
                            double* __restrict__ ax, double* __restrict__ ay, double* __restrict__ az,
                            double* __restrict__ bx, double* __restrict__ by, double* __restrict__ bz,
                            double* __restrict__ cx, double* __restrict__ cy, double* __restrict__ cz,
                            int Npart)
{
    int i = threadindex;
    if (i < Npart)
    {
        double3 a = make_double3(ax[i], ay[i], az[i]);
        double3 b = make_double3(bx[i], by[i], bz[i]);
        double3 c = make_double3(cx[i], cy[i], cz[i]);
        double dotP = dot(a, b);
        double sign = -dot(b, c); // !!!
        dst[i] = copysign(dotP, sign);
    }
}


__export__ void dotMaskAsync(double* dst, 
                             double* ax, double* ay, double* az, 
                             double* bx, double* by, double* bz, 
                             double axMul, double ayMul, double azMul,
                             double bxMul, double byMul, double bzMul,
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


__export__ void dotAsync(double* dst, double* ax, double* ay, double* az, double* bx, double* by, double* bz, CUstream stream, int Npart)
{
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    dotKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (dst, ax, ay, az, bx, by, bz, Npart);
}

__export__ void dotSignAsync(double* dst, double* ax, double* ay, double* az, double* bx, double* by, double* bz, double* cx, double* cy, double* cz, CUstream stream, int Npart)
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
