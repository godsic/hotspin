/*
  * @file
  *
  * @author Mykola Dvornik
  */

#ifndef _TS_H_
#define _TS_H_

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif

DLLEXPORT void tsAsync(float* Ts,
                              float* msat,
                              float* msat0T0,
                              float* Tc,
                              float* S,
                              const float msatMul,
                              const float msat0T0Mul,
                              const float TcMul,
                              const float SMul,
                              int Npart,
                              CUstream stream);

#ifdef __cplusplus
}
#endif

#endif
