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

DLLEXPORT  void llbar_nonlocal00nc_async(float* tx, float*  ty, float*  tz,

        float*  hx, float*  hy, float*  hz,

        float* msat0T0,

        float* lambda_e_xx,
        float* lambda_e_yy,
        float* lambda_e_zz,

        const float lambda_eMul_xx,
        const float lambda_eMul_yy,
        const float lambda_eMul_zz,

        const int sx, const int sy, const int sz,
        const float csx, const float csy, const float csz,
        const int pbc_x, const int pbc_y, const int pbc_z,
        CUstream stream);

#ifdef __cplusplus
}
#endif
#endif

