/*
  * @file
  * This file implements perpendicular Baryakhtar's relaxation
  * See: unpublished W Wang, ..., MD, VVK, MF, HFG (2012)
  *
  * @author Mykola Dvornik
  */

#ifndef _BARYAKHTAR_TORQUE_H_
#define _BARYAKHTAR_TORQUE_H_

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif

DLLEXPORT  void llbar_torque_async(double* tx, double*  ty, double*  tz,
                                   double*  Mx, double*  My, double*  Mz,
                                   double*  hx, double*  hy, double*  hz,

                                   double* msat0T0,

                                   CUstream stream,
                                   int Npart);

#ifdef __cplusplus
}
#endif
#endif

