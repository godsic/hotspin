/*
  * @file
  *
  * @author Mykola Dvornik
  */

#ifndef _ENERGY_FLOW_H_
#define _ENERGY_FLOW_H_

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif

DLLEXPORT void energyFlowAsync(float* w,
                          float* mx, float* my, float* mz,
                          float* Rx, float* Ry, float* Rz,
                          float* Tc,
                          float* S,
                          float* n,
                          float SMul,
                          int Npart,
                          CUstream stream);

#ifdef __cplusplus
}
#endif

#endif
