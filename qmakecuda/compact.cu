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

#include "compact_implement.h"

__global__ void
luminanceAsAlpha(unsigned char *od, int w)
{
    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;
    uint i = y * w + x;

    float luminance = 0.3 * (float)od[i*4+2] + 0.59 * (float)od[i*4+1] + 0.11 * (float)od[i*4];
    
    if((uint)luminance> 5)
        od[i*4+3] = (unsigned char)luminance;
    else
        od[i*4+3] = 0;

    //od[i*4] = 5; // Blue
    //od[i*4+1] = 55; //Green
    //od[i*4+2] = 255; //Red
    //od[i*4+3] = 255; // Alpha
    
}


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

    //od[i*4] = 5; // Blue
    //od[i*4+1] = 55; //Green
    //od[i*4+2] = 255; //Red
    //od[i*4+3] = 255; // Alpha
    
}

__global__ void
count_valid(unsigned char *color, uint *valid_in_group, uint group_size, uint limit)
{
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;
    uint count = 0;
    for(uint i = 0; i < group_size; i++)
    {
        uint pix_loc = pos * group_size + i;
        if(pix_loc < limit)
        {
            if(color[pix_loc*4+3] > 5)
                count++;
        }
    }
    
    valid_in_group[pos] = count;
}

__global__ void
compact_image(uint *prefix_sum, unsigned char *color, unsigned char *od, uint group_size, uint limit)
{
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;
    uint offset = prefix_sum[pos];
    for(uint i = 0; i < group_size; i++)
    {
        uint pix_loc = pos * group_size + i;
        if(pix_loc < limit)
        {   
            if(color[pix_loc*4+3] > 5)
            {
                od[offset*4] = color[pix_loc*4]; // blue
                od[offset*4+1] = color[pix_loc*4+1]; // green
                od[offset*4+2] = color[pix_loc*4+2]; // red
                od[offset*4+3] = 255; // alpha
                offset += 1;
            }
        }
    }
}


__global__ void
mergeScanGroups(uint *in, uint *element, uint *out)
{
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;
    uint offset = 0;
    uint i_group = pos/1024;
    for(uint i = 0; i < i_group; i++)
    {
        offset += in[i * 1024 + 1023] + element[i * 1024 + 1023];
        
    }
    out[pos] = in[pos] + offset;
}

__global__ void
black_image(unsigned char *out, int w)
{
    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;
    uint i = y * w + x;

    out[i*4] = out[i*4+1] = out[i*4+2] = 0;
    out[i*4+3] = 255;
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


extern "C" 
void countTexture(int width, int height, unsigned char *pImage, unsigned char *outImage)
{
    uint num_pixel = width * height;
    int size = num_pixel * sizeof(unsigned char) * 4;
// image in
    unsigned char *d_color;
    cutilSafeCall( cudaMalloc((void **)&d_color, size) );
    cutilSafeCall( cudaMemcpy(d_color, pImage, size, cudaMemcpyHostToDevice) );

// save luminance
    dim3 threadsPerBlock(8, 8, 1);
    dim3 blocksPerGrid(width / threadsPerBlock.x, height / threadsPerBlock.y, 1);
    luminanceAsAlpha<<<  blocksPerGrid, threadsPerBlock >>>(d_color, width);

// round up prefix_sum array size    
    uint num_buf = num_pixel;
    uint group_size = 32;
    uint round_size = group_size * 1024;
    if(num_buf % round_size > 0)
        num_buf += round_size - num_buf % round_size;

// divide pixels into groups
    uint num_group = num_buf / group_size;

// non-black counts per group    
    uint *d_count;
    cutilSafeCall( cudaMalloc((void **)&d_count, sizeof(uint) * num_group ) );

    count_valid<<<num_group/64, 64>>>(d_color, d_count, group_size, num_pixel);

    uint *d_scanResult;
    cutilSafeCall( cudaMalloc((void **)&d_scanResult, sizeof(uint) * num_group ));

    uint arrayLength = 1024; 
// prefix_sum               
    scanExclusive(d_scanResult, d_count, num_group / arrayLength, arrayLength);

    cutilSafeCall( cudaDeviceSynchronize() );

// merge scan groups
    uint *d_scanMerge;
    cudaMalloc((void **)&d_scanMerge, sizeof(uint) * num_group );
 
    mergeScanGroups<<<num_group/64, 64>>>(d_scanResult, d_count, d_scanMerge);

    //checkScanResult(d_scanMerge, d_count, num_group);

    unsigned char *d_out;
    cudaMalloc((void **)&d_out, size );
    black_image<<< blocksPerGrid, threadsPerBlock >>>(d_out, width);

// image out    
    compact_image<<<num_group/group_size, group_size>>>(d_scanMerge, d_color, d_out, group_size, num_pixel);
    cutilSafeCall( cudaMemcpy( outImage, d_out,  size, cudaMemcpyDeviceToHost));

    cutilSafeCall( cudaFree(d_color));
    cutilSafeCall( cudaFree(d_count));
    cutilSafeCall( cudaFree(d_scanResult));
    cutilSafeCall( cudaFree(d_scanMerge));
    cutilSafeCall( cudaFree(d_out));
}




#endif 