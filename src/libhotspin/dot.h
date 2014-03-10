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
DLLEXPORT void dotMaskAsync(double* dst, 
                             double* ax, double* ay, double* az, 
                             double* bx, double* by, double* bz, 
                             double axMul, double ayMul, double azMul,
                             double bxMul, double byMul, double bzMul,
                             CUstream stream, int Npart);

DLLEXPORT void dotAsync(double* dst, double* ax, double* ay, double* az, double* bx, double* by, double* bz, CUstream stream, int Npart);

/// calculates the dot product and takes the sign according to that of arguments, e.g. -1*-1= -1, 1*1=1
DLLEXPORT void dotSignAsync(double* dst, double* ax, double* ay, double* az, double* bx, double* by, double* bz, double* cx, double* cy, double* cz, CUstream stream, int Npart);

#ifdef __cplusplus
}
#endif
#endif
