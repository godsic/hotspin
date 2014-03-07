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
/// @param Npart number of floats per GPU, so total number of floats / nDevice()
DLLEXPORT void addAsync(float* dst, float* a, float* b, CUstream stream, int Npart);

/// dst[i] = MulA*a[i] + MulB*b[i]
/// @param Npart number of floats per GPU, so total number of floats / nDevice()
DLLEXPORT void linearCombination2Async(float* dst, float* a, float mulA, float* b, float mulB, CUstream stream, int NPart);

/// dst[i] = MulA*a[i] + MulB*b[i] + MulC*c[i]
/// @param Npart number of floats per GPU, so total number of floats / nDevice()
DLLEXPORT void linearCombination3Async(float* dst, float* a, float mulA, float* b, float mulB, float* c, float mulC, CUstream stream, int NPart);

/// dst[i] = a[i] + Mul * (b[i] + c[i])
/// @param Npart number of floats per GPU, so total number of floats / nDevice()
DLLEXPORT void addMaddAsync(float* dst, float* a, float* b, float* c, float mul, CUstream stream, int NPart);

/// Multiply-add: a[i] += mulB * b[i]
/// @param Npart number of floats per GPU, so total number of floats / nDevice()
DLLEXPORT void madd1Async(float* a, float* b, float mulB, CUstream stream, int Npart);

/// Multiply-add: a[i] += mulB * b[i] + mulC * c[i]
/// @param Npart number of floats per GPU, so total number of floats / nDevice()
DLLEXPORT void madd2Async(float* a, float* b, float mulB, float* c, float mulC, CUstream stream, int Npart);


/// Multiply-add: dst[i] = a[i] + mulB * b[i]
/// @param Npart number of floats per GPU, so total number of floats / nDevice()
DLLEXPORT void maddAsync(float* dst, float* a, float* b, float mulB, CUstream stream, int Npart);

/// Multiply-add: dst_j[i] = a_j[i] + mulB_j * b_j[i]
/// @param Npart number of floats per GPU, so total number of floats / nDevice()
DLLEXPORT void vecMaddAsync(float* dstx, float* dsty, float* dstz,
                            float* ax, float* ay, float* az,
                            float* bx, float* by, float* bz,
                            float mulBx, float mulBy, float mulBz,
                            CUstream stream, int Npart);

/// Complex multiply add: dst[i] += (a + bI) * kern[i] * src[i]
/// @param dst contains complex numbers (interleaved format)
/// @param src contains complex numbers (interleaved format)
/// @param kern contains real numbers
/// @param NComplexPart: number of complex numbers in dst per GPU (== number of real numbers in src per GPU)
///	dst[i] += c * src[i]
DLLEXPORT void cmaddAsync(float* dst, float a, float b, float* kern, float* src, CUstream stream, int NComplexPart);


#ifdef __cplusplus
}
#endif
#endif
