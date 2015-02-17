# general options to be included in all Makefiles
# options for DEBUG
# flags for the nvcc compiler

export LDFLAGS = $LDFLAGS:'./libmumax2'

NVCC=nvcc
  
NVCC+=\
  --compiler-options -Werror\
  --compiler-options -fPIC\
  --fmad=true\
  -gencode arch=compute_20,code=sm_20\
  -gencode arch=compute_20,code=sm_21\
  -gencode arch=compute_30,code=sm_30\
  -gencode arch=compute_35,code=sm_35\
  -gencode arch=compute_50,code=sm_50\
  -gencode arch=compute_52,code=sm_52\
  -Xptxas\
  -v\
  
CC=g++\

CC+=\
 -I/usr/local/cuda/include\
 -Wall\
 -O3