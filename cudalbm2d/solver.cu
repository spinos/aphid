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
#define TILE_I 8
#define TILE_J 8

texture<float, cudaTextureType2D> scalarTex;

__global__ void
show_scalar(float *id, unsigned char *od, int w, int h)
{
    uint x = blockIdx.x*TILE_I + threadIdx.x;
    uint y = blockIdx.y*TILE_J + threadIdx.y;
    uint i = y * w + x;

    float luminance = id[i] * 255;
    if(luminance > 255.f)
        luminance = 255.f;

    od[i*3] = od[i*3+1] = od[i*3+2] = (unsigned char)luminance;
}

__global__ void
advect_scalar(float *result, float *u, int w, int h)
{
    uint x = blockIdx.x*TILE_I + threadIdx.x;
    uint y = blockIdx.y*TILE_J + threadIdx.y;
    uint i = y * w + x;

    float tx = (float)x + 0.5 - u[i * 2];
    float ty = (float)y + 0.5 - u[i * 2 + 1];
    result[i] = tex2D(scalarTex, tx, ty);	
    if(result[i] < 0.f) result[i] = 0.f;
    if(result[i] > 1.5f) result[i] = 1.5f;
}

extern "C" 
void showScalarField(int width, int height, float *pScalar, unsigned char *outImage)
{
    const int size = width * height * sizeof(unsigned char) * 3;
	
    unsigned char *d_Out;
    cutilSafeCall( cudaMalloc((void **)&d_Out, size) );
    
    const int scalarSize = width * height * sizeof(float);
    float *d_In;
    cutilSafeCall( cudaMalloc((void **)&d_In, scalarSize) );
    cutilSafeCall( cudaMemcpy(d_In, pScalar, scalarSize, cudaMemcpyHostToDevice) );

    dim3 grid = dim3(width/TILE_I, height/TILE_J);
    dim3 block = dim3(TILE_I, TILE_J);
    
    show_scalar<<< grid, block>>>(d_In, d_Out, width, height);
	
    cutilSafeCall( cudaMemcpy( outImage, d_Out,  size, cudaMemcpyDeviceToHost));
    
    cutilSafeCall( cudaFree(d_In));
    cutilSafeCall( cudaFree(d_Out)); 

}

extern "C" 
void advectScalarField(int width, int height, float*u, float*field)
{
    const int fieldSize = width * height * sizeof(float);
    
    // allocate array and copy image data
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray* cu_array;
    cutilSafeCall( cudaMallocArray( &cu_array, &channelDesc, width, height )); 
    cutilSafeCall( cudaMemcpyToArray( cu_array, 0, 0, field, fieldSize, cudaMemcpyHostToDevice));

    scalarTex.addressMode[0] = cudaAddressModeClamp;
    scalarTex.addressMode[1] = cudaAddressModeClamp;
    scalarTex.filterMode = cudaFilterModeLinear;
    scalarTex.normalized = false;

    // Bind the array to the texture
    cutilSafeCall( cudaBindTextureToArray(scalarTex, cu_array, channelDesc));

    float *d_field;
    cutilSafeCall( cudaMalloc((void **)&d_field, fieldSize) );
    
    const int uSize = width * height * sizeof(float) * 2;
    float *d_u;
    cutilSafeCall( cudaMalloc((void **)&d_u, uSize) );
    cutilSafeCall( cudaMemcpy(d_u, u, uSize, cudaMemcpyHostToDevice) );
    
    dim3 grid = dim3(width/TILE_I, height/TILE_J);
    dim3 block = dim3(TILE_I, TILE_J);
    advect_scalar<<<grid, block>>>(d_field, d_u, width, height);
    
    cutilSafeCall( cudaMemcpy( field, d_field, fieldSize, cudaMemcpyDeviceToHost));
    
    cutilSafeCall(cudaUnbindTexture(scalarTex));
    cutilSafeCall(cudaFreeArray(cu_array));
    cutilSafeCall( cudaFree(d_field));
    cutilSafeCall( cudaFree(d_u));
}

#endif 