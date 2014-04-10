/*
  * @file
  * This file implements simple linear algebra functions.
  *
  * @author Arne Vansteenkiste
  */

#ifndef _MUL_H_
#define _MUL_H_

#include <cuda.h>
#include "cross_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

/// dst[i] = a[i] * b[i]
DLLEXPORT void mulAsync(double* dst, double* a, double* b, CUstream stream, int Npart);

DLLEXPORT void tensSYMMVecMul(double* dstX, double* dstY, double* dstZ,
						     double* srcX, double* srcY, double* srcZ,
						     double* kernXX, double* kernYY, double* kernZZ,
						     double* kernYZ, double* kernXZ, double* kernXY,
						     double srcMulX, double srcMulY, double srcMulZ,
						     int Nx, int Ny, int Nz,
						     CUstream stream);

#ifdef __cplusplus
}
#endif
#endif
