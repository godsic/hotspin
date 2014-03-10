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

DLLEXPORT  void llbar_local00nc_async(double* tx, double*  ty, double*  tz,
                                      double*  hx, double*  hy, double*  hz,

                                      double* msat0T0,

                                      double* lambda_xx,
                                      double* lambda_yy,
                                      double* lambda_zz,

                                      const double lambdaMul_xx,
                                      const double lambdaMul_yy,
                                      const double lambdaMul_zz,

                                      CUstream stream,
                                      int Npart);

#ifdef __cplusplus
}
#endif
#endif

