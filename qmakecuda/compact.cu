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
luminanceAsAlpha1(uchar4 *od, int w)
{
    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;
    uint i = y * w + x;
    float luminance = 0.3 * (float)od[i].z + 0.59 * (float)od[i].y + 0.11 * (float)od[i].x;
    
    if((uint)luminance> 5)
        od[i].w = (unsigned char)luminance;
    else
        od[i].w = 0;

//od[i*4+1] = 125;
//od[i*4+2] = 225;
//od[i*4+3] = 255;
    //float luminance = 0.3 * (float)od[i].z + 0.59 * (float)od[i].y + 0.11 * (float)od[i].x;
    
    //if((uint)luminance> 5)
    //    od[i].w = (unsigned char)luminance;
    //else
    //    od[i].w = 0;

    //od[i*4] = 5; // Blue
    //od[i*4+1] = 55; //Green
    //od[i*4+2] = 255; //Red
    //od[i*4+3] = 255; // Alpha
    
}

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
count_valid(uchar4 *color, uint *valid_in_group, uint group_size, uint limit)
{
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;
    uint count = 0;
    for(uint i = 0; i < group_size; i++)
    {
        uint pix_loc = pos * group_size + i;
        if(pix_loc < limit)
        {
            if(color[pix_loc].w > 5)
                count++;
        }
    }
    
    valid_in_group[pos] = count;
}

__global__ void
compact_image(uint *prefix_sum, uchar4 *color, uchar4 *od, uint *key, uint group_size, uint limit)
{
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;
    uint offset = prefix_sum[pos];
    for(uint i = 0; i < group_size; i++)
    {
        uint pix_loc = pos * group_size + i;
        if(pix_loc < limit)
        {   
            if(color[pix_loc].w > 5)
            {
                od[offset] = color[pix_loc];
                od[offset].w = 255;
                key[offset] = color[pix_loc].w; // alpha
                offset++;
            }
        }
    }
}




__global__ void
black_image(uchar4 *out, int w)
{
    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;
    uint i = y * w + x;

    out[i].x = out[i].y = out[i].z = out[i].w = 0;
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
void compactImage(int width, int height, unsigned char *pImage, unsigned char *outImage, char needSort)
{
     

    
    uint num_pixel = width * height;
    int size = num_pixel * sizeof(unsigned char) * 4;
// image in
    uchar4 *device_col;
    cutilSafeCall( cudaMalloc((void **)&device_col, sizeof(uchar4) * num_pixel ) );
    cudaMemcpy( device_col , pImage, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(8, 8, 1);
    dim3 blocksPerGrid(width / threadsPerBlock.x, height / threadsPerBlock.y, 1);
// save luminance
    luminanceAsAlpha1<<<  blocksPerGrid, threadsPerBlock >>>(device_col, width);

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

    count_valid<<<num_group/32, 32>>>(device_col, d_count, group_size, num_pixel);

    uint *d_scanResult;
    cutilSafeCall( cudaMalloc((void **)&d_scanResult, sizeof(uint) * num_group ));

    uint arrayLength = 1024; 
// prefix_sum               
    uint num_valid = scanExclusive(d_scanResult, d_count, num_group / arrayLength, arrayLength);

    cutilSafeCall( cudaDeviceSynchronize() );
    
    //checkScanResult(d_scanResult, d_count, num_group );
    
    printf("there are %i out of %i elements to sort\n", num_valid, num_pixel);
    printf("rate of compaction: %f percent\n", (1.f - (float)num_valid / (float)num_pixel ) * 100);


    uchar4 *device_out;
    cutilSafeCall( cudaMalloc((void **)&device_out, sizeof(uchar4) * num_pixel ));
    uint *device_key;
    cutilSafeCall( cudaMalloc((void **)&device_key, sizeof(uint) * num_pixel ));
    
// set to all empty, invalid elements will be invisible
    black_image<<<  blocksPerGrid, threadsPerBlock >>>(device_out, width);

// compact and store luminance as key to sort    
    compact_image<<<num_group/group_size, group_size>>>(d_scanResult, device_col, device_out, device_key, group_size, num_pixel);
  
    if(needSort)
    {
// sort by descending order    
    thrust::sort_by_key(thrust::device_ptr<uint>(device_key),
                        thrust::device_ptr<uint>(device_key) + num_valid,
                        thrust::device_ptr<uchar4>(device_out),
                        thrust::greater<uint>());
    }
    
// image out    
    cutilSafeCall( cudaMemcpy( outImage, thrust::raw_pointer_cast (device_out),  size, cudaMemcpyDeviceToHost));

    cutilSafeCall( cudaFree(device_col));
    cutilSafeCall( cudaFree(d_count));
    cutilSafeCall( cudaFree(d_scanResult));
    cutilSafeCall( cudaFree(device_out));
    cutilSafeCall( cudaFree(device_key));
}




#endif 