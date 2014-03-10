/*
  * @file
  * This file implements simple linear algebra functions.
  *
  * @author Arne Vansteenkiste
  */

#ifndef _WAVG_H_
#define _WAVG_H_

#include <cuda.h>
#include "cross_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

/// dst[i] = w0 * a[i]  + w1 * b[i] + R * sqrt(w0 * a[i] * w1 * b[i]) / (w0 + w1 + R * sqrt(w0*w1))
/// @param Npart number of doubles per GPU, so total number of doubles / nDevice()
DLLEXPORT void wavgAsync(double* dst, 
						 double* a, double* b, 
						 double* w0, double* w1, 
						 double* R,
						 double w0Mul,
						 double w1Mul,
						 double RMul,
						 CUstream stream, int Npart);

#ifdef __cplusplus
}
#endif
#endif
