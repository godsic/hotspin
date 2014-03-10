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
/// @param Npart number of doubles per GPU, so total number of doubles / nDevice()
DLLEXPORT void addHaniUniaxialAsync(double **hx, double **hy, double **hz,
                                    double **mx, double **my, double **mz,
                                    <<< <<< < HEAD
                                    double **Ku_map, double Ku_mul,
                                    double **anisU_mapx, double anisU_mulx,
                                    double **anisU_mapy, double anisU_muly,
                                    double **anisU_mapz, double anisU_mulz,
                                    == == == =
                                        double *anisK_map, double anisK_mul,
                                    double **anisAxes_map, double *anisAxes_mul,
                                    >>> >>> > anisotropy files added, not yet compiled
                                    CUstream* stream, int Npart
                                   );

/// dst[i] = dst[i] + hani(m[i])  in units [mSat]     cubic case
/// anisotropy constants devided by [mSat]
/// @param Npart number of doubles per GPU, so total number of doubles / nDevice()
DLLEXPORT void addHaniCubicAsync(double **hx, double **hy, double **hz,
                                 double **mx, double **my, double **mz,
                                 <<< <<< < HEAD
                                 double **K1_map, double K1_mul,
                                 double **K2_map, double K2_mul,
                                 double **anisU1_mapx, double anisU1_mulx,
                                 double **anisU1_mapy, double anisU1_muly,
                                 double **anisU1_mapz, double anisU1_mulz,
                                 double **anisU2_mapx, double anisU2_mulx,
                                 double **anisU2_mapy, double anisU2_muly,
                                 double **anisU2_mapz, double anisU2_mulz,
                                 == == == =
                                     double *anisK1_map, double anisK1_mul,
                                 double *anisK2_map, double anisK2_mul,
                                 double **anisAxes_map, double *anisAxes_mul,
                                 >>> >>> > anisotropy files added, not yet compiled
                                 CUstream* stream, int Npart
                                );




#ifdef __cplusplus
}
#endif
#endif
