/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

#include "gpu_conf.h"
#include "gpu_properties.h"
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

void check3dconf(dim3 gridSize, dim3 blockSize){

  //debugvv( printf("check3dconf((%d, %d, %d),(%d, %d, %d))\n", gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y, blockSize.z) );
  
  cudaDeviceProp* prop = (cudaDeviceProp*)gpu_getproperties();
  int maxThreadsPerBlock = prop->maxThreadsPerBlock;
  int* maxBlockSize = prop->maxThreadsDim;
  int* maxGridSize = prop->maxGridSize;
  
  assert(gridSize.x > 0);
  assert(gridSize.y > 0);
  assert(gridSize.z > 0);
  
  assert(blockSize.x > 0);
  assert(blockSize.y > 0);
  assert(blockSize.z > 0);
  
  assert((int)blockSize.x <= maxBlockSize[X]);
  assert((int)blockSize.y <= maxBlockSize[Y]);
  assert((int)blockSize.z <= maxBlockSize[Z]);
  
  assert((int)gridSize.x <= maxGridSize[X]);
  assert((int)gridSize.y <= maxGridSize[Y]);
  assert((int)gridSize.z <= maxGridSize[Z]);
  
  assert((int)(blockSize.x * blockSize.y * blockSize.z) <= maxThreadsPerBlock);
}

void check1dconf(int gridsize, int blocksize){
  assert(gridsize > 0);
  assert(blocksize > 0);
  assert(blocksize <= ((cudaDeviceProp*)gpu_getproperties())->maxThreadsPerBlock);
}

int _gpu_max_threads_per_block = 0;

int gpu_maxthreads(){
  if(_gpu_max_threads_per_block <= 0){
    cudaDeviceProp* prop = (cudaDeviceProp*)gpu_getproperties();
    _gpu_max_threads_per_block = prop->maxThreadsPerBlock;
  }
  return _gpu_max_threads_per_block;
}

void gpu_setmaxthreads(int max){
  _gpu_max_threads_per_block = max;
}

void make1dconf(int N, dim3* gridSize, dim3* blockSize){

  cudaDeviceProp* prop = (cudaDeviceProp*)gpu_getproperties();
  int maxBlockSize = gpu_maxthreads() / 2; // mad: initially we assumed it to be equal to gpu_maxthreads(), but using half of it improves gpu occupancy due to the resources pressure 
  int maxGridSize = prop->maxGridSize[X];

  (*blockSize).x = maxBlockSize;
  (*blockSize).y = 1;
  (*blockSize).z = 1;
  
  int N2 = divUp(N, maxBlockSize); // N2 blocks left
  
  int NX = divUp(N2, maxGridSize);
  int NY = divUp(N2, NX);

  (*gridSize).x = NX;
  (*gridSize).y = NY;
  (*gridSize).z = 1;

  assert((int)((*gridSize).x * (*gridSize).y * (*gridSize).z * (*blockSize).x * (*blockSize).y * (*blockSize).z) >= N);
  
  check3dconf(*gridSize, *blockSize);
}


void make2dconf(int N1, int N2, dim3* gridSize, dim3* blockSize){

#define BLOCKSIZE 16 ///@todo use device properties

	(*gridSize).x = divUp(N2, BLOCKSIZE);
	(*gridSize).y = divUp(N1, BLOCKSIZE);
	(*gridSize).z = 1;

	(*blockSize).x = BLOCKSIZE;
	(*blockSize).y = BLOCKSIZE;
	(*blockSize).z = 1;
  
  	check3dconf(*gridSize, *blockSize);
}

void make3dconf(int N0, int N1, int N2, dim3* gridSize, dim3* blockSize) {

	(*gridSize).x = divUp(N0, BLOCKSIZE3DX);
	(*gridSize).y = divUp(N1, BLOCKSIZE3DY);
	(*gridSize).z = divUp(N2, BLOCKSIZE3DZ);

	(*blockSize).x = BLOCKSIZE3DX;
	(*blockSize).y = BLOCKSIZE3DY;
	(*blockSize).z = BLOCKSIZE3DZ;
  
  	check3dconf(*gridSize, *blockSize);
}

#ifdef __cplusplus
}
#endif
