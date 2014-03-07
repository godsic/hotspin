/*
  * @file
  * This file implements heat flux density calculation
  *
  * @author Mykola Dvornik
  */

#ifndef _QINTER_H_
#define _QINTER_H_

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif


/// calculates heat flux densities for various subsystems

DLLEXPORT void QinterAsync(float* Qi,
                           float* Ti, float* Tj,
                           float* Gij,
                           float GijMul,
                           int Npart,
                           CUstream stream);

#ifdef __cplusplus
}
#endif
#endif
