# general options to be included in all Makefiles
# options for DEBUG
# flags for the nvcc compiler

export LDFLAGS = $LDFLAGS:'./libmumax2'

NVCC=nvcc
  
NVCC+=\
  --compiler-options -Werror\
  --compiler-options -fPIC\
  --use_fast_math\
  -gencode arch=compute_20,code=sm_20\
  -gencode arch=compute_20,code=sm_21\
  -gencode arch=compute_30,code=sm_30\
  -gencode arch=compute_35,code=sm_35\
  -Xptxas\
  -v\
  
# flags for the gcc compiler
# As CUDA is delivered only in 32 bits on Os X, architecture is forced to i386 on Darwin platform

CC=g++\

CC+=\
 -I/usr/local/cuda/include\
 -Wall\
 -O3