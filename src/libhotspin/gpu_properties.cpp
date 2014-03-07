/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

#include "gpu_properties.h"

#include "gpu_conf.h"
#include "gpu_safe.h"

#ifdef __cplusplus
extern "C" {
#endif


cudaDeviceProp* gpu_device_properties = NULL;

void* gpu_getproperties(void){
  if(gpu_device_properties == NULL){
    int device = -1;
    gpu_safe( cudaGetDevice(&device) );

    gpu_device_properties = (cudaDeviceProp*)malloc(sizeof(cudaDeviceProp));
    gpu_safe( cudaGetDeviceProperties(gpu_device_properties, device) );
  }
  return gpu_device_properties;
}


void gpu_print_properties(FILE* out){
  int device = -1;
  gpu_safe( cudaGetDevice(&device) );
  
  cudaDeviceProp prop;
  gpu_safe( cudaGetDeviceProperties(&prop, device) ); 
  
  int MiB = 1024 * 1024;
  int kiB = 1024;
  
  fprintf(out, "     Device number: %d\n", device);
  fprintf(out, "       Device name: %s\n", prop.name);
  fprintf(out, "     Global Memory: %d MiB\n", (int)(prop.totalGlobalMem/MiB));
  fprintf(out, "     Shared Memory: %d kiB/block\n", (int)(prop.sharedMemPerBlock/kiB));
  fprintf(out, "   Constant memory: %d kiB\n", (int)(prop.totalConstMem/kiB));
  fprintf(out, "         Registers: %d per block\n", (int)(prop.regsPerBlock/kiB));
  fprintf(out, "         Warp size: %d threads\n", (int)(prop.warpSize));
  //fprintf(out, "  Max memory pitch: %d bytes\n", (int)(prop.memPitch));
  fprintf(out, " Texture alignment: %d bytes\n", (int)(prop.textureAlignment));
  fprintf(out, " Max threads/block: %d\n", prop.maxThreadsPerBlock);
  fprintf(out, "    Max block size: %d x %d x %d threads\n", prop.maxThreadsDim[X], prop.maxThreadsDim[Y], prop.maxThreadsDim[Z]);
  fprintf(out, "     Max grid size: %d x %d x %d blocks\n", prop.maxGridSize[X], prop.maxGridSize[Y], prop.maxGridSize[Z]);
  fprintf(out, "Compute capability: %d.%d\n", prop.major, prop.minor);
  fprintf(out, "        Clock rate: %d MHz\n", prop.clockRate/1000);
  fprintf(out, "   Multiprocessors: %d\n", prop.multiProcessorCount);
  fprintf(out, "   Timeout enabled: %d\n", prop.kernelExecTimeoutEnabled);
  fprintf(out, "      Compute mode: %d\n", prop.computeMode);
  fprintf(out, "    Device overlap: %d\n", prop.deviceOverlap);
  fprintf(out, "Concurrent kernels: %d\n", prop.concurrentKernels);
  fprintf(out, "        Integrated: %d\n", prop.integrated);
  fprintf(out, "  Can map host mem: %d\n", prop.canMapHostMemory);
  
}


void gpu_print_properties_stdout(){
  gpu_print_properties(stdout);
}

#ifdef __cplusplus
}
#endif
