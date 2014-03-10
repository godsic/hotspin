/*
  * @file
  * This file implements transverse Baryakhtar's relaxation
  *
  * @author Mykola Dvornik
  */

#ifndef _LLBAR_LOCAL02NC_H_
#define _LLBAR_LOCAL02NC_H_

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif

    DLLEXPORT  void llbar_local02nc_async(double* tx, double*  ty, double*  tz,
                                          double*  mx, double*  my, double*  mz,
                                          double*  hx, double*  hy, double*  hz,

                                          double* msat0T0,

                                          double* mu_xx,
                                          double* mu_yy,
                                          double* mu_zz,

                                          const double muMul_xx,
                                          const double muMul_yy,
                                          const double muMul_zz,

                                          CUstream stream,
                                          int Npart);

#ifdef __cplusplus
}
#endif
#endif

