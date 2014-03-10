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
DLLEXPORT void decomposeAsync(double* Mx, double* My, double* Mz,
                              double* mx, double* my, double* mz,
                              double* msat,
                              double msatMul,
                              CUstream stream, int Npart);


#ifdef __cplusplus
}
#endif

#endif
