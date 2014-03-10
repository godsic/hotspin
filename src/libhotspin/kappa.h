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

DLLEXPORT void kappaAsync(double* kappa,
                          double* msat0,
                          double* msat0T0,
                          double* T,
                          double* Tc,
                          double* S,
                          double* n,
                          const double msat0Mul,
                          const double msat0T0Mul,
                          const double TcMul,
                          const double SMul,
                          int Npart,
                          CUstream stream);

#ifdef __cplusplus
}
#endif

#endif
