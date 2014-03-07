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

DLLEXPORT void cpAsync(float* Cp,
                          float* T,
                          float* Td,
                          float* n,
                          const float TdMul,
                          int Npart,
                          CUstream stream);

#ifdef __cplusplus
}
#endif

#endif
