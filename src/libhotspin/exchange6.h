/*
  * @file
  * This file implements 6-neighbor exchange
  *
  * @author Arne Vansteenkiste
  */

#ifndef _EXCHANGE6_H_
#define _EXCHANGE6_H_

#include <cuda.h>
#include "cross_platform.h"


#ifdef __cplusplus
extern "C" {
#endif


DLLEXPORT void exchange6Async(double* hx, double* hy, double* hz, 
                              double* mx, double* my, double* mz, 
                              double* msat0T0, 
                              double* lex, 
                              int N0, int N1Part, int N2, 
                              int periodic0, int periodic1, int periodic2,
                              double msat0T0Mul,
                              double lex2Mul_cellSizeX2, double lex2Mul_cellSizeY2, double lex2Mul_cellSizeZ2, 
                              CUstream streams);


#ifdef __cplusplus
}
#endif
#endif
