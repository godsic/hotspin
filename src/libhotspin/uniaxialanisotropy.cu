#include "uniaxialanisotropy.h"

#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"

#ifdef __cplusplus
extern "C" {
#endif

__global__ void uniaxialAnisotropyKern (double *hx, double *hy, double *hz,
                                        double *mx, double *my, double *mz,
                                        double *Ku_map, double* mSat_map, double Ku2_Mu0Msat_mul,
                                        double *anisU_mapx, double anisU_mulx,
                                        double *anisU_mapy, double anisU_muly,
                                        double *anisU_mapz, double anisU_mulz,
                                        int Npart)
{

    int i = threadindex;

    if (i < Npart)
    {

        double mSat_mask;
        if (mSat_map == NULL)
        {
            mSat_mask = 1.0;
        }
        else
        {
            mSat_mask = mSat_map[i];
            if (mSat_mask == 0.0)
            {
                mSat_mask = 1.0; // do not divide by zero
            }
        }

        double Ku2_Mu0Msat; // 2 * Ku / Mu0 * Msat
        if (Ku_map == NULL)
        {
            Ku2_Mu0Msat = Ku2_Mu0Msat_mul / mSat_mask;
        }
        else
        {
            Ku2_Mu0Msat = (Ku2_Mu0Msat_mul / mSat_mask) * Ku_map[i];
        }

        double ux;
        if (anisU_mapx == NULL)
        {
            ux = anisU_mulx;
        }
        else
        {
            ux = anisU_mulx * anisU_mapx[i];
        }

        double uy;
        if (anisU_mapy == NULL)
        {
            uy = anisU_muly;
        }
        else
        {
            uy = anisU_muly * anisU_mapy[i];
        }

        double uz;
        if (anisU_mapz == NULL)
        {
            uz = anisU_mulz;
        }
        else
        {
            uz = anisU_mulz * anisU_mapz[i];
        }

        double mu = mx[i] * ux + my[i] * uy + mz[i] * uz;
        hx[i] = Ku2_Mu0Msat * mu * ux;
        hy[i] = Ku2_Mu0Msat * mu * uy;
        hz[i] = Ku2_Mu0Msat * mu * uz;
    }

}



__export__ void uniaxialAnisotropyAsync(double *hx, double *hy, double *hz,
                                        double *mx, double *my, double *mz,
                                        double *Ku1_map, double *MSat_map, double Ku2_Mu0Msat_mul,
                                        double *anisU_mapx, double anisU_mulx,
                                        double *anisU_mapy, double anisU_muly,
                                        double *anisU_mapz, double anisU_mulz,
                                        CUstream stream, int Npart)
{

    dim3 gridSize, blockSize;
    make1dconf(Npart, &gridSize, &blockSize);

    uniaxialAnisotropyKern <<< gridSize, blockSize, 0, cudaStream_t(stream)>>> (
        hx, hy, hz,
        mx, my, mz,
        Ku1_map, MSat_map, Ku2_Mu0Msat_mul,
        anisU_mapx, anisU_mulx,
        anisU_mapy, anisU_muly,
        anisU_mapz, anisU_mulz,
        Npart);
}

#ifdef __cplusplus
}
#endif
