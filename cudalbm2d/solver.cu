#ifndef _RADIXSORT_KERNEL_H_
#define _RADIXSORT_KERNEL_H_

#include <cuda_runtime.h>


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/detail/type_traits.h>

#include <cutil_inline.h>

#include <shrUtils.h>
#include <algorithm>
#include <time.h>
#include <limits.h>

#include "solver_implement.h"

__global__ void
get_alpha(unsigned char *id, unsigned char *od, int w, int h)
{
    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;
    uint i = y * w + x;

    float luminance = 0.3 * (float)id[i*4+2] + 0.59 * (float)id[i*4+1] + 0.11 * (float)id[i*4];
    
    if((uint)luminance> 5)
        od[i] = (unsigned char)luminance;
    else
        od[i] = 0;
    
}

extern "C" 
void initTexture(int width, int height, unsigned char *pImage, unsigned char *outImage)
{
    int size = width * height * sizeof(unsigned char) * 4;
	
    unsigned char *d_Input, *d_tmp;
	cutilSafeCall( cudaMalloc((void **)&d_Input, size) );
	cutilSafeCall( cudaMalloc((void **)&d_tmp, size) );
	cutilSafeCall( cudaMemcpy(d_Input, pImage, size, cudaMemcpyHostToDevice) );

    get_alpha<<< width*height/16, 16>>>(d_Input, d_tmp, width, height);
	
    cutilSafeCall( cudaMemcpy( outImage, d_tmp,  size, cudaMemcpyDeviceToHost));
    cutilSafeCall( cudaFree(d_Input));
    cutilSafeCall( cudaFree(d_tmp)); 

}

#endif 