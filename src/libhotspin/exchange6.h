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


DLLEXPORT void exchange6Async(float* hx, float* hy, float* hz, 
                              float* mx, float* my, float* mz, 
                              float* msat0T0, 
                              float* lex, 
                              int N0, int N1Part, int N2, 
                              int periodic0, int periodic1, int periodic2, 
                              float lex2Mulmsat0T0Mul_cellSizeX2, float lex2Mulmsat0T0Mul_cellSizeY2, float lex2Mulmsat0T0Mul_cellSizeZ2, 
                              CUstream streams);


#ifdef __cplusplus
}
#endif
#endif
