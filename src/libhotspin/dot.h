/*
  * @file
  * This file implements the torque according to Landau-Lifshitz.
  *
  * @author Arne Vansteenkiste
  */

#ifndef _DOT_H_
#define _DOT_H_

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif


/// calculates the dot product
DLLEXPORT void dotMaskAsync(float* dst, 
                             float* ax, float* ay, float* az, 
                             float* bx, float* by, float* bz, 
                             float axMul, float ayMul, float azMul,
                             float bxMul, float byMul, float bzMul,
                             CUstream stream, int Npart);

DLLEXPORT void dotAsync(float* dst, float* ax, float* ay, float* az, float* bx, float* by, float* bz, CUstream stream, int Npart);

/// calculates the dot product and takes the sign according to that of arguments, e.g. -1*-1= -1, 1*1=1
DLLEXPORT void dotSignAsync(float* dst, float* ax, float* ay, float* az, float* bx, float* by, float* bz, float* cx, float* cy, float* cz, CUstream stream, int Npart);

#ifdef __cplusplus
}
#endif
#endif
