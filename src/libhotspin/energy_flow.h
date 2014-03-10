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

DLLEXPORT void energyFlowAsync(double* w,
                          double* mx, double* my, double* mz,
                          double* Rx, double* Ry, double* Rz,
                          double* Tc,
                          double* S,
                          double* n,
                          double SMul,
                          int Npart,
                          CUstream stream);

#ifdef __cplusplus
}
#endif

#endif
