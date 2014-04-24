/*
  * @file
  * This file implements common functions typically required for calculus.
  *
  *
  * @author Mykola Dvornik
  */

#ifndef _COMMON_FUNC_H_
#define _COMMON_FUNC_H_


#include <cuda.h>
#include <math_constants.h>
#include "stdio.h"

// printf() is only supported
// for devices of compute capability 2.0 and higher
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif


#define kB          1.380650424E-23                               // Boltzmann's constant in J/K
#define muB         9.2740091523E-24                              // Bohr magneton in Am^2
#define mu0         4.0 * 1e-7 * CUDART_PI                        // vacuum permeability
#define zero        1.0e-32                                       // the zero threshold
#define PI4         CUDART_PI * CUDART_PI * CUDART_PI * CUDART_PI // PI^4
#define eps         1.0e-8                                        // the target numerical accuracy of iterative methods
#define linRange    1.0e-1                                        // Defines the region of linearity
#define INTMAXSTEPS 61                                            // Defines maximum amount of steps for numerical integration    
#define INFINITESPINLIMIT 1.0e5                                   // Above this value the spin is treated as infinite (classical)
  
typedef double (*func)(double x, double prefix, double mult);
typedef double (*funcTs)(double x, double prefix, double mult, double C);

// Python-like modulus
inline __host__ __device__ int Mod(int a, int b)
{
    return (a % b + b) % b;
}

inline __host__ __device__ double dot(double3 a, double3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}


inline __host__ __device__ double3 cross(double3 a, double3 b)
{
    return make_double3( - a.y * b.z + a.z * b.y, - a.z * b.x + a.x * b.z,  - a.x * b.y + a.y * b.x);
}

inline __host__ __device__ double len(double3 a)
{
    return sqrt(dot(a, a));
}

inline __host__ __device__ double3 normalize(double3 a)
{
    double veclen = (len(a) != 0.0) ? 1.0 / len(a) : 0.0;
    return make_double3(a.x * veclen, a.y * veclen, a.z * veclen);
}

inline __device__ double coth(double x)
{
    return 1.0 / tanh(x);
}


inline __device__ double Bj(double J, double x)
{
    double lpre = 1.0 / (2.0 * J);
    double gpre = (2.0 * J + 1.0) * lpre;
    double gpre2 = gpre * gpre;
    double lpre2 = lpre * lpre;
    double lpre4 = lpre2 * lpre2;
    double gpre4 = gpre2 * gpre2;
    double lpre6 = lpre4 * lpre2;
    double gpre6 = gpre4 * gpre2;
    double lpre8 = lpre4 * lpre4;
    double gpre8 = gpre4 * gpre4;
    double limA = linRange / gpre;
    double limB = linRange / lpre;
    double lim = fmax(limA, limB);
    return (fabs(x) < lim)  ? ((gpre2 - lpre2) * x / 3.0) - 
                              ((gpre4 - lpre4) * x * x * x / 45.0) + 
                              ((gpre6 - lpre6) * x * x * x * x * x * 2.0 / 945.0) - 
                              ((gpre8 - lpre8) * x * x * x * x * x  * x * x / 4725.0)
                            : gpre * coth(gpre * x) - lpre * coth(lpre * x);
}

inline __device__ double dBjdx(double J, double x)
{
    double lpre = 1.0 / (2.0 * J);
    double gpre = (2.0 * J + 1.0) * lpre;
    double gpre2 = gpre * gpre;
    double lpre2 = lpre * lpre;
    double lpre4 = lpre2 * lpre2;
    double gpre4 = gpre2 * gpre2;
    double lpre6 = lpre4 * lpre2;
    double gpre6 = gpre4 * gpre2;
    double lpre8 = lpre4 * lpre4;
    double gpre8 = gpre4 * gpre4;
    double limA = linRange / gpre;
    double limB = linRange / lpre;
    double lim = fmax(limA, limB);
    return (fabs(x) < lim) ? ((gpre2 - lpre2) / 3.0) - 
                             ((gpre4 - lpre4) * x * x / 15.0) + 
                             ((gpre6 - lpre6) * x * x * x * x * 2.0 / 189.0) - 
                             ((gpre8 - lpre8) * x * x * x * x  * x * x / 675.0)
           : (gpre2 - lpre2) + lpre2 * coth(lpre * x) * coth(lpre * x) - gpre2 * coth(gpre * x) * coth(gpre * x);
}

inline __device__ double L(double x)
{
    return (fabs(x) < linRange) ? (x / 3.0) -
                                  (x * x * x / 45.0) +
                                  (x * x * x * x * x * 2.0 / 945.0) -
                                  (x * x * x * x * x  * x * x / 4725.0)
                                : coth(x) - (1.0 / x) ;
}

inline __device__ double dLdx(double x)
{
    return (fabs(x) < linRange) ? (1 / 3.0) - 
                                  (x * x / 15.0) +
                                  (x * x * x * x * 2.0 / 189.0) -
                                  (x * x * x * x  * x * x / 675.0)
                                : 1.0 - (coth(x) * coth(x)) + (1.0 / (x * x));
}

inline __device__ double sign(double x)
{
    double val = (signbit(x) == 0.0) ? 1.0 : -1.0;
    return (x == 0.0) ? 0.0 : val;
}

inline __device__ double getMaskUnity(double *msk, int idx)
{
    return (msk == NULL) ? 1.0 : msk[idx];
}

inline __device__ double getMaskZero(double *msk, int idx)
{
    return (msk == NULL) ? 0.0 : msk[idx];
}

inline __device__ double fdivZero(double a, double b)
{
    return (b == 0.0) ? 0.0 : a / b ;
}

inline __device__ double avgGeomZero(double a, double b)
{
    double a_b = a + b;
    return (a_b == 0.0) ? 0.0 : 2.0 * a * b / a_b;
}

inline __device__ double weightedAvgZero(double x0, double x1, double w0, double w1, double R)
{
    double denom = w0 + w1 + 2.0 * R * sqrt(w0 * w1);
    return (denom == 0.0) ? 0.0 : (w0 * x0 + w1 * x1 + 2.0 * R * sqrt(w0 * x0 * w1 * x1)) / denom;
}

inline __device__ double Debye(double x) 
{ 
    double nom = x * x * x * x * exp(x);
    double denom = (exp(x) - 1.0) * (exp(x) - 1.0);
    return (x <= zero) ? 0.0 : nom / denom;
}

inline __device__ float Debyef(float x) 
{ 
    float nom = x * x * x * x * __expf(x);
    float denom = (__expf(x) - 1.0f) * (__expf(x) - 1.0f);
    return (x <= zero) ? 0.0f : nom / denom;
}

#endif
