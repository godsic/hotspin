/*
  * @file
  * This file implements simple linear algebra functions.
  *
  * @author Arne Vansteenkiste
  */

#ifndef _ADD_H_
#define _ADD_H_

#include <cuda.h>
#include "cross_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

/// dst[i] = a[i] + b[i]
/// @param Npart number of doubles per GPU, so total number of doubles / nDevice()
DLLEXPORT void addAsync(double* dst, double* a, double* b, CUstream stream, int Npart);

/// dst[i] = MulA*a[i] + MulB*b[i]
/// @param Npart number of doubles per GPU, so total number of doubles / nDevice()
DLLEXPORT void linearCombination2Async(double* dst, double* a, double mulA, double* b, double mulB, CUstream stream, int NPart);

/// dst[i] = MulA*a[i] + MulB*b[i] + MulC*c[i]
/// @param Npart number of doubles per GPU, so total number of doubles / nDevice()
DLLEXPORT void linearCombination3Async(double* dst, double* a, double mulA, double* b, double mulB, double* c, double mulC, CUstream stream, int NPart);

/// dst[i] = a[i] + Mul * (b[i] + c[i])
/// @param Npart number of doubles per GPU, so total number of doubles / nDevice()
DLLEXPORT void addMaddAsync(double* dst, double* a, double* b, double* c, double mul, CUstream stream, int NPart);

/// Multiply-add: a[i] += mulB * b[i]
/// @param Npart number of doubles per GPU, so total number of doubles / nDevice()
DLLEXPORT void madd1Async(double* a, double* b, double mulB, CUstream stream, int Npart);

/// Multiply-add: a[i] += mulB * b[i] + mulC * c[i]
/// @param Npart number of doubles per GPU, so total number of doubles / nDevice()
DLLEXPORT void madd2Async(double* a, double* b, double mulB, double* c, double mulC, CUstream stream, int Npart);


/// Multiply-add: dst[i] = a[i] + mulB * b[i]
/// @param Npart number of doubles per GPU, so total number of doubles / nDevice()
DLLEXPORT void maddAsync(double* dst, double* a, double* b, double mulB, CUstream stream, int Npart);

/// Multiply-add: dst_j[i] = a_j[i] + mulB_j * b_j[i]
/// @param Npart number of doubles per GPU, so total number of doubles / nDevice()
DLLEXPORT void vecMaddAsync(double* dstx, double* dsty, double* dstz,
                            double* ax, double* ay, double* az,
                            double* bx, double* by, double* bz,
                            double mulBx, double mulBy, double mulBz,
                            CUstream stream, int Npart);

/// Complex multiply add: dst[i] += (a + bI) * kern[i] * src[i]
/// @param dst contains complex numbers (interleaved format)
/// @param src contains complex numbers (interleaved format)
/// @param kern contains real numbers
/// @param NComplexPart: number of complex numbers in dst per GPU (== number of real numbers in src per GPU)
///	dst[i] += c * src[i]
DLLEXPORT void cmaddAsync(double* dst, double a, double b, double* kern, double* src, CUstream stream, int NComplexPart);


#ifdef __cplusplus
}
#endif
#endif
