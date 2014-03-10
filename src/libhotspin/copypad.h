	/*
  * @file
  *
  * @author Arne Vansteenkiste
  */

#ifndef _COPYPAD_H_
#define _COPYPAD_H_

#include <cuda.h>
#include "cross_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Copy+zero pad a 3D matrix
/// @param dst: destination arrays
/// @param D0: dst X size, >= S0
/// @param D1: dst Y size, >= S1
/// @param D2: dst Z size, >= S2
/// @param src: source arrays
/// @param S0: source X size , <= D0
/// @param S1: source Y size , <= D1
/// @param S2: source Z size , <= D2
/// @param Ncomp: number of array components

DLLEXPORT void copyPad3DAsync(double* dst, int D0, int D1, int D2, double* src, int S0, int S1, int S2, int Ncomp, CUstream streams);

DLLEXPORT void copyUnPad3DAsync(double* dst, int D0, int D1, int D2, double* src, int S0, int S1, int S2, int Ncomp, CUstream streams);

/// Put an array to zero with (sub)sizes [NO, N1part, N2]
DLLEXPORT void zeroArrayAsync(double *A, int N, CUstream streams);

#ifdef __cplusplus
}
#endif
#endif
