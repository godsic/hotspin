/*
  * @file
  * This file implements perpendicular Baryakhtar's relaxation
  * See: unpublished W Wang, ..., MD, VVK, MF, HFG (2012)
  *
  * @author Mykola Dvornik
  */

#ifndef _LLBAR_NONLOCAL00NC_H_
#define _LLBAR_NONLOCAL00NC_H_

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif

DLLEXPORT  void llbar_nonlocal00nc_async(double* tx, double*  ty, double*  tz,

        double*  hx, double*  hy, double*  hz,

        double* msat0T0,

        double* lambda_e_xx,
        double* lambda_e_yy,
        double* lambda_e_zz,

        const double lambda_eMul_xx,
        const double lambda_eMul_yy,
        const double lambda_eMul_zz,

        const int sx, const int sy, const int sz,
        const double csx, const double csy, const double csz,
        const int pbc_x, const int pbc_y, const int pbc_z,
        CUstream stream);

#ifdef __cplusplus
}
#endif
#endif

