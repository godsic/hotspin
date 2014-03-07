/*
  * @file
  * This file implements the uniaxial anisotropy field
  *
  * @author Ben Van de Wiele, Arne Vansteenkiste
  */

#ifndef _UNIAXIALANISOTROPY_
#define _UNIAXIALANISOTROPY_

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif

/// @param Npart number of floats per GPU, so total number of floats / nDevice()
DLLEXPORT void uniaxialAnisotropyAsync(float *hx, float *hy, float *hz,
                                       float *mx, float *my, float *mz,
                                       float *Ku1_map, float *MSat_map, float Ku2_Mu0Msat_mul,
                                       float *anisU_mapx, float anisU_mulx,
                                       float *anisU_mapy, float anisU_muly,
                                       float *anisU_mapz, float anisU_mulz,
                                       CUstream stream, int Npart);

#ifdef __cplusplus
}
#endif
#endif
