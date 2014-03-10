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

DLLEXPORT void tsAsync(double* Ts,
                              double* msat,
                              double* msat0T0,
                              double* Tc,
                              double* S,
                              const double msatMul,
                              const double msat0T0Mul,
                              const double TcMul,
                              const double SMul,
                              int Npart,
                              CUstream stream);

#ifdef __cplusplus
}
#endif

#endif
