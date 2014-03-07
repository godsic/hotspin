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
/// @param Npart number of floats per GPU, so total number of floats / nDevice()
DLLEXPORT void wavgAsync(float* dst, 
						 float* a, float* b, 
						 float* w0, float* w1, 
						 float* R,
						 float w0Mul,
						 float w1Mul,
						 float RMul,
						 CUstream stream, int Npart);

#ifdef __cplusplus
}
#endif
#endif
