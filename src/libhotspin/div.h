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
/// @param Npart number of doubles per GPU, so total number of doubles / nDevice()
DLLEXPORT void divAsync(double* dst, double* a, double* b, CUstream stream, int Npart);

#ifdef __cplusplus
}
#endif
#endif
