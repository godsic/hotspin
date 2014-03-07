/*
  * @file
  * This file implements simple linear algebra functions.
  *
  * @author Mykola Dvornik
  */

#ifndef _DIV_H_
#define _DIV_H_

#include <cuda.h>
#include "cross_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

/// dst[i] = a[i] / b[i]
/// @param Npart number of floats per GPU, so total number of floats / nDevice()
DLLEXPORT void divAsync(float* dst, float* a, float* b, CUstream stream, int Npart);

#ifdef __cplusplus
}
#endif
#endif
