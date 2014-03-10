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

/// @param Npart number of doubles per GPU, so total number of doubles / nDevice()
DLLEXPORT void uniaxialAnisotropyAsync(double *hx, double *hy, double *hz,
                                       double *mx, double *my, double *mz,
                                       double *Ku1_map, double *MSat_map, double Ku2_Mu0Msat_mul,
                                       double *anisU_mapx, double anisU_mulx,
                                       double *anisU_mapy, double anisU_muly,
                                       double *anisU_mapz, double anisU_mulz,
                                       CUstream stream, int Npart);

#ifdef __cplusplus
}
#endif
#endif
