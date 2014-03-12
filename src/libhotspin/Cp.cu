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
                          const int Npart)
{

    int i = threadindex;

    if (i < Npart)
    {
        const double n = getMaskUnity(nMsk, i);
        const double Td = TdMul * getMaskUnity(TdMsk, i);
        const double T = Tp[i];

        if (T == 0.0 || Td == 0.0 || n == 0.0)
        {
            Cp[i] = 0.0;
            return;
        }

        double xx = Td / T;

        const double h = xx / (double)INTMAXSTEPS;
        const double h_2 = 0.5 * h;

        double x = 0.0;  
        double val = Debye(x);
        double valm = 0.0;

        int j = 0;

        for (j=0; j < INTMAXSTEPS-1; j++) {
            x += h;
            valm += (Debye(x));
            
        }

        x += h;
        val += Debye(x);
        val += (2.0 * valm);
        val *= h_2;

        Cp[i] = n * val / (xx * xx * xx); // kb, nMul should be accounted in the upstream multiplier

    }
}

__export__ void cpAsync(double* Cp,
                          double* T,
                          double* Td,
                          double* n,
                          const double TdMul,
                          const int Npart,
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
