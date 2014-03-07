/*
  * @file
  * This file implements longitudinal recovery field of exchange nature
  * See
  *
  * @author Mykola Dvornik,  Arne Vansteenkiste
  */

#ifndef _LONG_FIELD_H
#define _LONG_FIELD_H

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif

DLLEXPORT void long_field_async(float* hx, float* hy, float* hz,
                                 float* mx, float* my, float* mz,
                                 float* msat0T0,
                                 float* S,
                                 float* n,
                                 float* Tc,
                                 float* Ts,
                                 float msat0T0Mul,
                                 float SMul,
                                 float nMul,
                                 float TcMul,
                                 float TsMul,
                                 int NPart,
                                 CUstream stream);

#ifdef __cplusplus
}
#endif
#endif
