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
#include "stdio.h"

// printf() is only supported
// for devices of compute capability 2.0 and higher
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif


#define kB          1.380650424E-23f                    // Boltzmann's constant in J/K
#define muB         9.2740091523E-24f                   // Bohr magneton in Am^2
#define mu0         4.0f * 1e-7f * 3.14159265358979f    // vacuum permeability
#define zero        1.0e-32f                            // the zero threshold
#define eps         1.0e-8f                             // the target numerical accuracy of iterative methods
#define linRange    1.0e-1f                             // Defines the region of linearity
#define linRangeD   1.0e-1                              // Defines the region of linearity
#define INTMAXSTEPS 10000                               // Defines maximum amount of steps for numerical integration    

typedef float (*func)(float x, float prefix, float mult);
typedef double (*funcD)(double x, double prefix, double mult);
typedef float (*funcTs)(float x, float prefix, float mult, float C);
typedef double (*funcDTs)(double x, double prefix, double mult, double C);

typedef double real;

struct float5
{
    float x;
    float y;
    float z;
    float w;
    float v;
};

typedef struct float5 float5;

struct int6
{
    int x;
    int y;
    int z;
    int w;
    int v;
    int t;
};

typedef struct int6 int6;

struct real5
{
    real x;
    real y;
    real z;
    real w;
    real v;
};

typedef struct real5 real5;

struct real6
{
    real x;
    real y;
    real z;
    real w;
    real v;
    real t;
};

typedef struct real6 real6;

struct real7
{
    real x;
    real y;
    real z;
    real w;
    real v;
    real t;
    real q;
};

typedef struct real7 real7;

struct real4
{
    real x;
    real y;
    real z;
    real w;
};

typedef struct real4 real4;

struct real3
{
    real x;
    real y;
    real z;
};

typedef struct real3 real3;

inline __host__ __device__ int6 make_int6(int x, int y, int z, int w, int v, int t)
{
    int6 a;
    a.x = x;
    a.y = y;
    a.z = z;
    a.w = w;
    a.v = v;
    a.t = t;
    return a;
}

inline __host__ __device__ float5 make_float5(float x, float y, float z, float w, float v)
{
    float5 a;
    a.x = x;
    a.y = y;
    a.z = z;
    a.w = w;
    a.v = v;
    return a;
}

inline __host__ __device__ real7 make_real7(real x, real y, real z, real w, real t, real q)
{
    real7 a;
    a.x = x;
    a.y = y;
    a.z = z;
    a.w = w;
    a.t = t;
    a.q = q;
    return a;
}

inline __host__ __device__ real5 make_real5(real x, real y, real z, real w, real v)
{
    real5 a;
    a.x = x;
    a.y = y;
    a.z = z;
    a.w = w;
    a.v = v;
    return a;
}

inline __host__ __device__ real6 make_real6(real x, real y, real z, real w, real v, real t)
{
    real6 a;
    a.x = x;
    a.y = y;
    a.z = z;
    a.w = w;
    a.v = v;
    a.t = t;
    return a;
}

inline __host__ __device__ real3 make_real3(real x, real y, real z)
{
    real3 a;
    a.x = x;
    a.y = y;
    a.z = z;
    return a;
}

// Python-like modulus
inline __host__ __device__ int Mod(int a, int b)
{
    return (a % b + b) % b;
}

// dot product
inline __host__ __device__ float dotf(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ double dot(double3 a, double3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// cross product in LHR system

inline __host__ __device__ float3 crossf(float3 a, float3 b)
{
    return make_float3( - a.y * b.z + a.z * b.y,  - a.z * b.x + a.x * b.z, - a.x * b.y + a.y * b.x);
}

inline __host__ __device__ real3 cross(real3 a, real3 b)
{
    return make_real3( - a.y * b.z + a.z * b.y, - a.z * b.x + a.x * b.z,  - a.x * b.y + a.y * b.x);
}

// lenght of the 3-components vector
inline __host__ __device__ float len(float3 a)
{
    return sqrtf(dotf(a, a));
}

inline __host__ __device__ double len(double3 a)
{
    return sqrt(dot(a, a));
}

// normalize the 3-components vector
inline __host__ __device__ float3 normalizef(float3 a)
{
    float veclen = (len(a) != 0.0f) ? 1.0f / len(a) : 0.0f;
    return make_float3(a.x * veclen, a.y * veclen, a.z * veclen);
}

inline __host__ __device__ double3 normalize(double3 a)
{
    real veclen = (len(a) != 0.0) ? 1.0 / len(a) : 0.0;
    return make_double3(a.x * veclen, a.y * veclen, a.z * veclen);
}

inline __device__ float cothf(float x)
{
    return 1.0f / tanhf(x);
}

inline __device__ double coth(double x)
{
    return 1.0 / tanh(x);
}

/*inline __device__ double Bj(double J, double x)
{
    double lpre = 1.0 / (2.0 * J);
    double gpre = (2.0 * J + 1.0) * lpre;
    double lim = linRangeD / gpre;
    return (fabs(x) < lim) ? ((gpre * gpre - lpre * lpre) * x / 3.0) + ((pow(lpre, 4.0) - pow(gpre, 4.0)) * pow(x, 2.0) / 45.0) + (0.5 * (pow(gpre, 6.0) - pow(lpre, 6.0)) * pow(x, 5.0) / 945.0) + ((pow(gpre, 8.0) - pow(lpre, 8.0)) * pow(x, 7.0) / 4725.0)
           : gpre * coth(gpre * x) - lpre * coth(lpre * x);
}*/

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
    double lim = linRangeD / gpre;
    return (fabs(x) < lim)  ? ((gpre2 - lpre2) * x / 3.0) + 
                              ((lpre4 - gpre4) * x * x * x / 45.0) + 
                              ((gpre6 - lpre6) * x * x * x * x * x * 0.5 / 945.0) + 
                              ((gpre8 - lpre8) * x * x * x * x * x  * x * x / 4725.0)
                            : gpre * coth(gpre * x) - lpre * coth(lpre * x);
}

inline __device__ float Bjf(float J, float x)
{
    float lpre = 1.0f / (2.0f * J);
    float gpre = (2.0f * J + 1.0f) * lpre;
    float lim = linRange / gpre;
    float gpre2 = gpre * gpre;
    float lpre2 = lpre * lpre;
    float lpre4 = lpre2 * lpre2;
    float gpre4 = gpre2 * gpre2;
    float lpre6 = lpre4 * lpre2;
    float gpre6 = gpre4 * gpre2;
    float lpre8 = lpre4 * lpre4;
    float gpre8 = gpre4 * gpre4;
    return (fabsf(x) < lim) ? ((gpre2 - lpre2) * x / 3.0f) + 
                              ((lpre4 - gpre4) * x * x * x / 45.0f) + 
                              ((gpre6 - lpre6) * x * x * x * x * x * 0.5f / 945.0f) + 
                              ((gpre8 - lpre8) * x * x * x * x * x  * x * x / 4725.0f)
           : gpre * cothf(gpre * x) - lpre * cothf(lpre * x);
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
    double lim = linRange / gpre;
    return (fabs(x) < lim) ? ((gpre2 - lpre2) / 3.0) + 
                             ((lpre4 - gpre4) * x * x / 15.0) + 
                             ((gpre6 - lpre6) * x * x * x * x * 0.5 / 189.0) + 
                             ((gpre8 - lpre8) * x * x * x * x  * x * x / 675.0)
           : (gpre2 - lpre2) + lpre2 * coth(lpre * x) * coth(lpre * x) - gpre2 * coth(gpre * x) * coth(gpre * x);
}

inline __device__ float dBjdxf(float J, float x)
{
    float lpre = 1.0f / (2.0f * J);
    float gpre = (2.0f * J + 1.0f) * lpre;
    float gpre2 = gpre * gpre;
    float lpre2 = lpre * lpre;
    float lpre4 = lpre2 * lpre2;
    float gpre4 = gpre2 * gpre2;
    float lpre6 = lpre4 * lpre2;
    float gpre6 = gpre4 * gpre2;
    float lpre8 = lpre4 * lpre4;
    float gpre8 = gpre4 * gpre4;
    float lim = linRange / gpre;
    return (fabsf(x) < lim) ? ((gpre2 - lpre2) / 3.0f) + 
                              ((lpre4 - gpre4) * x * x / 15.0f) + 
                              ((gpre6 - lpre6) * x * x * x * x * 0.5f / 189.0f) + 
                              ((gpre8 - lpre8) * x * x * x * x  * x * x / 675.0f)
           : (gpre2 - lpre2) + lpre2 * cothf(lpre * x) * cothf(lpre * x) - gpre2 * cothf(gpre * x) * cothf(gpre * x);
}


inline __device__ float L(float x)
{
    return (x > -linRange && x < linRange) ? (x / 3.0f) - ((x * x * x) / 45.0f) : cothf(x) - (1.0f / x) ;
}

inline __device__ float dLdx(float x)
{
    return (x > -linRange && x < linRange) ? (1 / 3.0f) - ((x * x) / 15.0f) : 1.0f - (cothf(x) * cothf(x)) + (1.0f / (x * x));
}

// Classical function
inline __device__ float signf(float x)
{
    float val = (signbit(x) == 0.0f) ? 1.0f : -1.0f;
    return (x == 0.0f) ? 0.0f : val;
}

inline __device__ double sign(double x)
{
    double val = (signbit(x) == 0.0) ? 1.0 : -1.0;
    return (x == 0.0) ? 0.0 : val;
}

inline __device__ float getMaskUnity(float *msk, int idx)
{
    return (msk == NULL) ? 1.0f : msk[idx];
}

inline __device__ float getMaskZero(float *msk, int idx)
{
    return (msk == NULL) ? 0.0f : msk[idx];
}

inline __device__ float fdivZero(float a, float b)
{
    return (b == 0.0f) ? 0.0f : a / b ;
}

inline __device__ float avgGeomZero(float a, float b)
{
    float a_b = a + b;
    return (a_b == 0.0f) ? 0.0f : 2.0f * a * b / a_b;
}

inline __device__ float weightedAvgZero(float x0, float x1, float w0, float w1, float R)
{
    float denom = w0 + w1 + 2.0f * R * sqrtf(w0 * w1);
    return (denom == 0.0f) ? 0.0f : (w0 * x0 + w1 * x1 + 2.0f * R * sqrtf(w0 * x0 * w1 * x1)) / denom;
}

inline __device__ float Debye(float x) 
{
    float nom = x * x * x * x * expf(x);
    float denom = (expf(x) - 1.0f) * (expf(x) - 1.0f);
    return (x <= zero) ? 0.0f : nom / denom; 
}

#endif
