/*
  * @file
  *
  * @author Mykola Dvornik
  */

#ifndef _DECOMPOSE_H_
#define _DECOMPOSE_H_

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif


/// Normalizes a vector array.
/// @param mx, my, mz: Components of vector array to normalize
/// @param norm_map: desired norm, may contain NULL pointers
DLLEXPORT void decomposeAsync(float* Mx, float* My, float* Mz,
                              float* mx, float* my, float* mz,
                              float* msat,
                              float msatMul,
                              CUstream stream, int Npart);


#ifdef __cplusplus
}
#endif

#endif
