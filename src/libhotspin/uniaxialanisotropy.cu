#include "uniaxialanisotropy.h"

#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"

#ifdef __cplusplus
extern "C" {
#endif

__global__ void uniaxialAnisotropyKern (float *hx, float *hy, float *hz,
                                        float *mx, float *my, float *mz,
                                        float *Ku_map, float* mSat_map, float Ku2_Mu0Msat_mul,
                                        float *anisU_mapx, float anisU_mulx,
                                        float *anisU_mapy, float anisU_muly,
                                        float *anisU_mapz, float anisU_mulz,
                                        int Npart)
{

    int i = threadindex;

    if (i < Npart)
    {

        float mSat_mask;
        if (mSat_map == NULL)
        {
            mSat_mask = 1.0f;
        }
        else
        {
            mSat_mask = mSat_map[i];
            if (mSat_mask == 0.0f)
            {
                mSat_mask = 1.0f; // do not divide by zero
            }
        }

        float Ku2_Mu0Msat; // 2 * Ku / Mu0 * Msat
        if (Ku_map == NULL)
        {
            Ku2_Mu0Msat = Ku2_Mu0Msat_mul / mSat_mask;
        }
        else
        {
            Ku2_Mu0Msat = (Ku2_Mu0Msat_mul / mSat_mask) * Ku_map[i];
        }

        float ux;
        if (anisU_mapx == NULL)
        {
            ux = anisU_mulx;
        }
        else
        {
            ux = anisU_mulx * anisU_mapx[i];
        }

        float uy;
        if (anisU_mapy == NULL)
        {
            uy = anisU_muly;
        }
        else
        {
            uy = anisU_muly * anisU_mapy[i];
        }

        float uz;
        if (anisU_mapz == NULL)
        {
            uz = anisU_mulz;
        }
        else
        {
            uz = anisU_mulz * anisU_mapz[i];
        }

        float mu = mx[i] * ux + my[i] * uy + mz[i] * uz;
        hx[i] = Ku2_Mu0Msat * mu * ux;
        hy[i] = Ku2_Mu0Msat * mu * uy;
        hz[i] = Ku2_Mu0Msat * mu * uz;
    }

}



__export__ void uniaxialAnisotropyAsync(float *hx, float *hy, float *hz,
                                        float *mx, float *my, float *mz,
                                        float *Ku1_map, float *MSat_map, float Ku2_Mu0Msat_mul,
                                        float *anisU_mapx, float anisU_mulx,
                                        float *anisU_mapy, float anisU_muly,
                                        float *anisU_mapz, float anisU_mulz,
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
