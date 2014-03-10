#include "Qinter.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "stdio.h"
#include <cuda.h>
#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal
__global__ void QinterKern(double* __restrict__ Qi,
                           const double* __restrict__ Ti, const double* __restrict__ Tj,
                           const double* __restrict__ GijMsk,
                           const double GijMul,
                           int Npart)
{

    int i = threadindex;
    if (i < Npart)
    {
        double Tii = Ti[i];
        double Tjj = Tj[i];
        double Gij = (GijMsk == NULL) ? GijMul : GijMul * GijMsk[i];
        Qi[i] = Gij * (Tjj - Tii);
    }
}

__export__ void QinterAsync(double* Qi,
                            double* Ti, double* Tj,
                            double* Gij,
                            double GijMul,
                            int Npart,
                            CUstream stream)
{
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    QinterKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (Qi,
            Ti,
            Tj,
            Gij,
            GijMul,
            Npart);
}

#ifdef __cplusplus
}
#endif

