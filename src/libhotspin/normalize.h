/*
  * @file
  *
  * @author Arne Vansteenkiste
  */

#ifndef _NORMALIZE_H_
#define _NORMALIZE_H_

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif


/// Normalizes a vector array.
/// @param mx, my, mz: Components of vector array to normalize
/// @param norm_map: desired norm, may contain NULL pointers
DLLEXPORT void normalizeAsync(float* mx, float* my, float* mz, 
							  CUstream stream, int Npart);


#ifdef __cplusplus
}
#endif

#endif
