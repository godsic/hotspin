/**
  * @file
  * This file implements the addition of the anisotropy field (uniaxial and cubic) to the effective field.
  *
  * @author Ben Van de Wiele
  */

#ifndef _ADD_HANI_
#define _ADD_HANI_

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif

/// dst[i] = dst[i] + hani(m[i]) in units [mSat]      uniaxial case
/// anisotropy constants devided by [mSat]
/// @param Npart number of floats per GPU, so total number of floats / nDevice()
DLLEXPORT void addHaniUniaxialAsync(float **hx, float **hy, float **hz,
                                    float **mx, float **my, float **mz,
                                    <<< <<< < HEAD
                                    float **Ku_map, float Ku_mul,
                                    float **anisU_mapx, float anisU_mulx,
                                    float **anisU_mapy, float anisU_muly,
                                    float **anisU_mapz, float anisU_mulz,
                                    == == == =
                                        float *anisK_map, float anisK_mul,
                                    float **anisAxes_map, float *anisAxes_mul,
                                    >>> >>> > anisotropy files added, not yet compiled
                                    CUstream* stream, int Npart
                                   );

/// dst[i] = dst[i] + hani(m[i])  in units [mSat]     cubic case
/// anisotropy constants devided by [mSat]
/// @param Npart number of floats per GPU, so total number of floats / nDevice()
DLLEXPORT void addHaniCubicAsync(float **hx, float **hy, float **hz,
                                 float **mx, float **my, float **mz,
                                 <<< <<< < HEAD
                                 float **K1_map, float K1_mul,
                                 float **K2_map, float K2_mul,
                                 float **anisU1_mapx, float anisU1_mulx,
                                 float **anisU1_mapy, float anisU1_muly,
                                 float **anisU1_mapz, float anisU1_mulz,
                                 float **anisU2_mapx, float anisU2_mulx,
                                 float **anisU2_mapy, float anisU2_muly,
                                 float **anisU2_mapz, float anisU2_mulz,
                                 == == == =
                                     float *anisK1_map, float anisK1_mul,
                                 float *anisK2_map, float anisK2_mul,
                                 float **anisAxes_map, float *anisAxes_mul,
                                 >>> >>> > anisotropy files added, not yet compiled
                                 CUstream* stream, int Npart
                                );




#ifdef __cplusplus
}
#endif
#endif
