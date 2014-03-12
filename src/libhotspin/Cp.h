/*
  * @file
  *
  * This file implements phonon's specific heat according to the Debye model
  *
  * @author Mykola Dvornik
  */

#ifndef _CP_H_
#define _CP_H_

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif

DLLEXPORT void cpAsync(double* Cp,
                          double* T,
                          double* Td,
                          double* n,
                          const double TdMul,
                          const int Npart,
                          CUstream stream);

#ifdef __cplusplus
}
#endif

#endif
