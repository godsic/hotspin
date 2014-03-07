/*
  * @file
  *
  * @author Arne Vansteenkiste
  */

#ifndef _TEMPERATURE_H_
#define _TEMPERATURE_H_

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif


DLLEXPORT void temperature_scaleAnizNoise(float* hx, float* hy, float* hz,
        float* mu_xx,
        float* mu_yy,
        float* mu_zz,
        float* tempMask,
        float* msat0T0Mask,

        float muMul_xx,
        float muMul_yy,
        float muMul_zz,

        float KB2tempMul_mu0VgammaDtMSatMul,
        CUstream stream,
        int Npart);
#ifdef __cplusplus
}
#endif
#endif
