/*
  * @file
  * This file implements transverse Baryakhtar's relaxation
  *
  * @author Mykola Dvornik
  */

#ifndef _LLBAR_LOCAL02C_H_
#define _LLBAR_LOCAL02C_H_

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif

DLLEXPORT  void llbar_local02c_async(float* tx, float*  ty, float*  tz,
                                     float*  mx, float*  my, float*  mz,
                                     float*  hx, float*  hy, float*  hz,

                                     float* msat0T0,

                                     float* mu_xx,
                                     float* mu_yy,
                                     float* mu_zz,

                                     const float muMul_xx,
                                     const float muMul_yy,
                                     const float muMul_zz,

                                     CUstream stream,
                                     int Npart);

#ifdef __cplusplus
}
#endif
#endif

