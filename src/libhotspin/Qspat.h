/*
  * @file
  * This file implements heat flux density calculation
  *
  * @author Mykola Dvornik
  */

#ifndef _QSPAT_H_
#define _QSPAT_H_

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif


/// calculates heat flux density caused by spatial temperature gradient

DLLEXPORT void Qspat_async(float* Q,
                           float* T,
                           float* k,
                           const float kMul,
                           const int sx, const int sy, const int sz,
                           const float csx, const float csy, const float csz,
                           const int pbc_x, const int pbc_y, const int pbc_z,
                           CUstream stream);

#ifdef __cplusplus
}
#endif
#endif
