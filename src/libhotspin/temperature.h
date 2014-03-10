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


DLLEXPORT void temperature_scaleAnizNoise(double* hx, double* hy, double* hz,
        double* mu_xx,
        double* mu_yy,
        double* mu_zz,
        double* tempMask,
        double* msat0T0Mask,

        double muMul_xx,
        double muMul_yy,
        double muMul_zz,

        double KB2tempMul_mu0VgammaDtMSatMul,
        CUstream stream,
        int Npart);
#ifdef __cplusplus
}
#endif
#endif
