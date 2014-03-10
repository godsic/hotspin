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

DLLEXPORT void long_field_async(double* hx, double* hy, double* hz,
                                 double* mx, double* my, double* mz,
                                 double* msat0T0,
                                 double* S,
                                 double* n,
                                 double* Tc,
                                 double* Ts,
                                 double msat0T0Mul,
                                 double SMul,
                                 double nMul,
                                 double TcMul,
                                 double TsMul,
                                 int NPart,
                                 CUstream stream);

#ifdef __cplusplus
}
#endif
#endif
