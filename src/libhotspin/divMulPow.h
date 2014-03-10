/*
  * @file
  * This file implements simple linear algebra functions.
  *
  * @author Mykola Dvornik
  */

#ifndef _DIVMULPOW_H_
#define _DIVMULPOW_H_

#include <cuda.h>
#include "cross_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

/// dst[i] = pow(c[i], p) * a[i] / b[i]
/// @param Npart number of doubles per GPU, so total number of doubles / nDevice()
DLLEXPORT void divMulPowAsync(double* dst, double* a, double* b, double* c, double p, CUstream stream, int Npart);

#ifdef __cplusplus
}
#endif
#endif
