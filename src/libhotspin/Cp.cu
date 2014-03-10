#include "Cp.h"

#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"

#ifdef __cplusplus
extern "C" {
#endif

///@internal

__global__ void cpKern(double* __restrict__ Cp,
                          double* __restrict__ Tp,
                          double* __restrict__ TdMsk,
                          double* __restrict__ nMsk,
                          const double TdMul,
                          int Npart)
{

    int i = threadindex;

    if (i < Npart)
    {
        double n = getMaskUnity(nMsk, i);
        double Td = TdMul * getMaskUnity(TdMsk, i);
        double T = Tp[i];

        if (T == 0.0 || Td == 0.0 || n == 0.0)
        {
            Cp[i] = 0.0;
            return;
        }

        double xx = Td / T;

        double h = xx / (double)INTMAXSTEPS;
        double h_2 = 0.5 * h;

        double x = 0.0;  
        double val = 0.0;

        while (x < xx) {
            val += (h_2 * (Debye(x) + Debye(x+h)));
            x += h;
        }

//         int i = 0;

// #pragma unroll 2
//         for (i = 0; i < INTMAXSTEPS; i++) {
//             val += (Debye(x) * h);
//             x += h;
//         }

        Cp[i] = 9.0 * n * val / (xx * xx * xx); // kb, nMul should be accounted in the upstream multiplier

    }
}

__export__ void cpAsync(double* Cp,
                          double* T,
                          double* Td,
                          double* n,
                          const double TdMul,
                          int Npart,
                          CUstream stream)
{
    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);
    cpKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (Cp,
            T,
            Td,
            n,
            TdMul,
            Npart);

}

#ifdef __cplusplus
}
#endif
