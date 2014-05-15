#include "normalize.h"

#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal
__global__ void normalizeKern(double* mx, double* my, double* mz,
                              int Npart)
{
    int i = threadindex;

    if (i < Npart)
    {
        // reconstruct norm from map

        double Mx = mx[i];
        double My = my[i];
        double Mz = mz[i];

        double Mnorm = sqrt(Mx * Mx + My * My + Mz * Mz);

        Mnorm = (Mnorm == 0.0) ? 0.0 : 1.0 / Mnorm;
        
        mx[i] = Mx * Mnorm;
        my[i] = My * Mnorm;
        mz[i] = Mz * Mnorm;
    }
}


__export__ void normalizeAsync(double* mx, double* my, double* mz, CUstream stream, int Npart)
{
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    normalizeKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (mx, my, mz, Npart);
}

#ifdef __cplusplus
}
#endif
