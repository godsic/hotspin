# general options to be included in all Makefiles
# options for DEBUG
# flags for the nvcc compiler

export LDFLAGS = $LDFLAGS:'./libmumax2'

ifndef SystemRoot

NVCC=nvcc
  
NVCC+=\
  --compiler-options -Werror\
  --compiler-options -fPIC\
  --use_fast_math\
  -gencode arch=compute_20,code=sm_20\
  -gencode arch=compute_20,code=sm_21\
  -gencode arch=compute_30,code=sm_30\
  -Xptxas\
  -v\
  
NVCCWERRORLESS=nvcc
NVCCWERRORLESS+=\
  --compiler-options -fPIC\
  --use_fast_math\
  -gencode arch=compute_20,code=sm_20\
  -gencode arch=compute_20,code=sm_21\
  -gencode arch=compute_30,code=sm_30\
  -Xptxas\
  -v\

# flags for the gcc compiler
# As CUDA is delivered only in 32 bits on Os X, architecture is forced to i386 on Darwin platform

CC=g++\

CC+=\
 -I/usr/local/cuda/include\
 -Wall\
 -O3\

 
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin) # Os X
CC+=\
 -arch i386
export GOARCH=386

else # Linux and other Unix based systems

CC+=\

endif

else

# On Win32 CUDA's NVCC supports only MSVC 05,08,10
# It is possible to use bleeding edge MSVC 11, with slightly tweaked host.h file...

# CCBIN = "C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin\cl.exe"

NVCC=nvcc --use-local-env --cl-version 2010
NVCC+=\
  -gencode arch=compute_20,code=sm_20\
  -gencode arch=compute_20,code=sm_21\
  -gencode arch=compute_30,code=sm_30\
  -Xptxas\
  -v\
  --use_fast_math\
  --compiler-options -W4\
  --compiler-options -Ox\
  --compiler-options -Oi\
  --compiler-options -EHsc\
  --compiler-options -GS\
  --compiler-options -fp:fast\
  --compiler-options -DWIN32\
  --compiler-options -DNDEBUG\
  --compiler-options -D_WINDOWS\
  
NVCCWERRORLESS=nvcc --use-local-env --cl-version 2010
NVCCWERRORLESS+=\
  -gencode arch=compute_20,code=sm_20\
  -gencode arch=compute_20,code=sm_21\
  -gencode arch=compute_30,code=sm_30\
  -Xptxas\
  -v\
  --use_fast_math\
  --compiler-options -W4\
  --compiler-options -Ox\
  --compiler-options -Oi\
  --compiler-options -EHsc\
  --compiler-options -GS\
  --compiler-options -fp:fast\
  --compiler-options -DWIN32\
  --compiler-options -DNDEBUG\
  --compiler-options -D_WINDOWS\

CC=cl.exe\

CC+=\
 /I "$(CUDA_INC_PATH)"\
 /DWIN32\
 /DNDEBUG\
 /D_WINDOWS\
 /W4\
 /Ox\
 /Oi\
 /EHsc\
 /GS\
 /fp:fast\
 
 endif
