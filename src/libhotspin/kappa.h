/*
  * @file
  *
  * @author Mykola Dvornik
  */

#ifndef _KAPPA_H_
#define _KAPPA_H_

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif

DLLEXPORT void kappaAsync(float* kappa,
                          float* msat0,
                          float* msat0T0,
                          float* T,
                          float* Tc,
                          float* S,
                          float* n,
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
