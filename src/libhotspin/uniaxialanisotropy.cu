#include "uniaxialanisotropy.h"

#include <cuda.h>
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "common_func.h"

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

        double msat = getMaskUnity(mSat_map, i);
        if (msat == 0.0)
        {
            hx[i] = 0.0;
            hy[i] = 0.0;
            hz[i] = 0.0;
            return;
        }

        double Ku = getMaskUnity(Ku_map, i);
        double Ku2_Mu0Msat = (Ku2_Mu0Msat_mul / msat) * Ku; // 2 * Ku / Mu0 * Msat

        double ux = anisU_mulx * getMaskUnity(anisU_mapx, i);
        double uy = anisU_muly * getMaskUnity(anisU_mapy, i);
        double uz = anisU_mulz * getMaskUnity(anisU_mapz, i);

        double3 m = make_double3(mx[i], my[i], mz[i]);
        double3 u = normalize(make_double3(ux, uy, uz));

        double mu = dot(m,u);

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
