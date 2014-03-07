/*
  * @file
  * This file implements perpendicular Baryakhtar's relaxation
  * See: unpublished W Wang, ..., MD, VVK, MF, HFG (2012)
  *
  * @author Mykola Dvornik
  */

#ifndef _LLBAR_LOCAL00NC_H_
#define _LLBAR_LOCAL00NC_H_

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif

DLLEXPORT  void llbar_local00nc_async(float* tx, float*  ty, float*  tz,
                                      float*  hx, float*  hy, float*  hz,

                                      float* msat0T0,

                                      float* lambda_xx,
                                      float* lambda_yy,
                                      float* lambda_zz,

                                      const float lambdaMul_xx,
                                      const float lambdaMul_yy,
                                      const float lambdaMul_zz,

                                      CUstream stream,
                                      int Npart);

#ifdef __cplusplus
}
#endif
#endif

