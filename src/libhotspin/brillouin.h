/*
  * @file
  *
  * @author Mykola Dvornik
  */

#ifndef _BRILLOUIN_H_
#define _BRILLOUIN_H_

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif

DLLEXPORT void brillouinAsync(float* msat0,
                              float* msat0T0,
                              float* T,
                              float* Tc,
                              float* S,
                              const float msat0Mul,
                              const float msat0T0Mul,
                              const float TcMul,
                              const float SMul,
                              int Npart,
                              CUstream stream);

#ifdef __cplusplus
}
#endif

#endif
