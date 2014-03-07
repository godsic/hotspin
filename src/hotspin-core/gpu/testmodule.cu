/*
 * Module to test CUDA module loading and execution.
 * To be compiled with:
 * nvcc -ptx testmodule.cu
 */


#ifdef __cplusplus
extern "C" {
#endif

#define threadindex ( ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x )

/// Sets the first N elements of array to value.
__global__ void testMemset(float value, float* array, int N){
	int i = threadindex;
	if(i < N){
		array[i] = value;
	}
}

/// DEBUG: used to find out the type names from PTX assembly.
__global__ void testTypes(int INT, long long int INT64, float FLOAT, double DOUBLE, float* FLOATPTR, void* VOIDPTR){

}


#ifdef __cplusplus
}
#endif
