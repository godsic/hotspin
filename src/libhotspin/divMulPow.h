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
/// @param Npart number of floats per GPU, so total number of floats / nDevice()
DLLEXPORT void divMulPowAsync(float* dst, float* a, float* b, float* c, float p, CUstream stream, int Npart);

#ifdef __cplusplus
}
#endif
#endif
