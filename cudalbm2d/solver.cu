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

#define TILE_DIM 16

const float visc = 0.009f;

float *f0_data, *f1_data, *f2_data, *f3_data, *f4_data, *f5_data, *f6_data, *f7_data, *f8_data;
float *density_data, *injection_data, *velocity_data;



texture<float, cudaTextureType2D> f1_tex, f2_tex, f3_tex, f4_tex,
                  f5_tex, f6_tex, f7_tex, f8_tex;
                  
cudaArray *f1_array, *f2_array, *f3_array, *f4_array, 
                *f5_array, *f6_array, *f7_array, *f8_array;
                
cudaArray *density_array;
texture<float, cudaTextureType2D> densityTex;

__global__ void
show_scalar(float *id, unsigned char*mask, unsigned char *od, int w, int h)
{
    uint x = blockIdx.x*TILE_DIM + threadIdx.x;
    uint y = blockIdx.y*TILE_DIM + threadIdx.y;
    uint i = y * w + x;
    
    if(mask[i] == 0) {
        od[i*3] = od[i*3+1] = od[i*3+2] = 0;
    }
    else {
        float luminance = id[i] * 255;
        if(luminance > 255.f)
        luminance = 255.f;

        od[i*3] = od[i*3+1] = od[i*3+2] = (unsigned char)luminance;
    }
}

__global__ void
show_velocity(float *id, unsigned char*mask, unsigned char *od, int w, int h)
{
    uint x = blockIdx.x*TILE_DIM + threadIdx.x;
    uint y = blockIdx.y*TILE_DIM + threadIdx.y;
    uint i = y * w + x;
    
    if(mask[i] == 0) {
        od[i*3] = od[i*3+1] = od[i*3+2] = 0;
    }
    else {
        float r = id[i*2] * 128 + 127;
        if(r > 255.f)
            r = 255.f;
        
        float g = id[i*2 + 1] * 128 + 127;
        if(g > 255.f)
            g = 255.f;
    
        od[i*3] = r;
        od[i*3+1] = g;
        od[i*3+2] = 127;
    }
}

__global__ void
advect_density(float *result, float *u, int w, int h)
{
    uint x = blockIdx.x*TILE_DIM + threadIdx.x;
    uint y = blockIdx.y*TILE_DIM + threadIdx.y;
    uint i = y * w + x;

    float tx = (float)x + 0.5 - u[i * 2];
    float ty = (float)y + 0.5 - u[i * 2 + 1];
    result[i] = tex2D(densityTex, tx, ty);	
    if(result[i] < 0.f) result[i] = 0.f;
    if(result[i] > 1.5f) result[i] = 1.25f;
}

__global__ void
inject_energy(float *d, 
              float *f1_data, float *f2_data, float *f3_data, float *f4_data,
              float *f5_data, float *f6_data, float *f7_data, float *f8_data,
              float *stir, unsigned char * obstacle, int w, int h)
{
    uint x = blockIdx.x*TILE_DIM + threadIdx.x;
    uint y = blockIdx.y*TILE_DIM + threadIdx.y;
    uint i = y * w + x;
    
    if(obstacle[i] > 0) {

        float ux = stir[i * 2];
        float uy = stir[i * 2 + 1];
        
        float e = ux * ux + uy * uy;
        d[i] += e;
    
        if(d[i] > 1.15f) d[i] = 1.15f;

        f1_data[i] += e * uy/9.f;
        f2_data[i] += e * ux/9.f;
        f3_data[i] += e * (-uy)/9.f;
        f4_data[i] += e * (-ux)/9.f;
        f5_data[i] += e * (ux + uy)/36.f;
        f6_data[i] += e * (ux - uy)/36.f;
        f7_data[i] += e * (-ux - uy)/36.f;
        f8_data[i] += e * (-ux + uy)/36.f;
    }
    
    stir[i * 2] *= 0.99f;
    stir[i * 2 + 1] *= 0.99f;
}

__global__ void
boundary_condition_kernel(float *f1_data, float *f2_data, float *f3_data, float *f4_data,
              float *f5_data, float *f6_data, float *f7_data, float *f8_data,
              unsigned char * obstacle, int w)
{
    uint x = blockIdx.x*TILE_DIM + threadIdx.x;
    uint y = blockIdx.y*TILE_DIM + threadIdx.y;
    uint i = y * w + x;

    float tmp;
    if(obstacle[i] < 1) {
        tmp = f2_data[i];
        f2_data[i] = f4_data[i];
        f4_data[i] = tmp;

        tmp = f1_data[i];
        f1_data[i] = f3_data[i];
        f3_data[i] = tmp;

        tmp = f8_data[i];
        f8_data[i] = f6_data[i];
        f6_data[i] = tmp;

        tmp = f7_data[i];
        f7_data[i] = f5_data[i];
        f5_data[i] = tmp;
    }   
}

__global__ void stream_kernel (float *f1_data, float *f2_data,
                               float *f3_data, float *f4_data, float *f5_data,
			       float *f6_data, float *f7_data, float *f8_data,
			       int width)
{
    uint x = blockIdx.x*TILE_DIM + threadIdx.x;
    uint y = blockIdx.y*TILE_DIM + threadIdx.y;
    uint i = y * width + x;

    f1_data[i] = tex2D(f1_tex, (float) x  , (float) (y-1));
    f2_data[i] = tex2D(f2_tex, (float) (x-1)  , (float) y);
    f3_data[i] = tex2D(f3_tex, (float) x  , (float) (y+1));
    f4_data[i] = tex2D(f4_tex, (float) (x+1)  , (float) y);
    f5_data[i] = tex2D(f5_tex, (float) (x-1)  , (float) (y-1));
    f6_data[i] = tex2D(f6_tex, (float) (x-1)  , (float) (y+1));
    f7_data[i] = tex2D(f7_tex, (float) (x+1)  , (float) (y+1));
    f8_data[i] = tex2D(f8_tex, (float) (x+1)  , (float) (y-1));
   
}

__global__ void reset_kernel (float *f0_data, float *f1_data, float *f2_data, float *f3_data, 
                                float *f4_data, float *f5_data, float *f6_data, float *f7_data, float *f8_data, 
                                float *density, float * injection, float * velocity, int width)
{
    uint x = blockIdx.x*TILE_DIM + threadIdx.x;
    uint y = blockIdx.y*TILE_DIM + threadIdx.y;
    uint i = y * width + x;
    
    float faceq1 = 4.f/9.f;
    float faceq2 = 1.f/9.f;
    float faceq3 = 1.f/36.f;
  
    f0_data[i] = faceq1;
    f1_data[i] = faceq2;
    f2_data[i] = faceq2;
    f3_data[i] = faceq2;
    f4_data[i] = faceq2;
    f5_data[i] = faceq3;
    f6_data[i] = faceq3;
    f7_data[i] = faceq3;
    f8_data[i] = faceq3;
    density[i] = 0.f;
    injection[i*2] = injection[i*2+1] = 0.f;
    velocity[i*2] = velocity[i*2+1] = 0.f;
}

__global__ void collide_kernel (float *f0_data, float *f1_data, float *f2_data, float *f3_data, 
                                float *f4_data, float *f5_data, float *f6_data, float *f7_data, float *f8_data, 
                                float *velocity, float * density, int width)
{
    uint x = blockIdx.x*TILE_DIM + threadIdx.x;
    uint y = blockIdx.y*TILE_DIM + threadIdx.y;
    uint i = y * width + x;
    float ro, v_x, v_y;
    float tau = (5.f*visc + 1.0f)/2.0f;
    float f0now, f1now, f2now, f3now, f4now, f5now, f6now, f7now, f8now;
    float f0eq, f1eq, f2eq, f3eq, f4eq, f5eq, f6eq, f7eq, f8eq; 

    // Read all f's and store in registers
    f0now = f0_data[i];
    f1now = f1_data[i];
    f2now = f2_data[i];
    f3now = f3_data[i];
    f4now = f4_data[i];
    f5now = f5_data[i];
    f6now = f6_data[i];
    f7now = f7_data[i];
    f8now = f8_data[i];
    
    float faceq1 = 4.f/9.f;
    float faceq2 = 1.f/9.f;
    float faceq3 = 1.f/36.f;

    // Macroscopic flow props:
    ro =  f0now + f1now + f2now + f3now + f4now + f5now + f6now + f7now + f8now;
    
    v_x = (f2now + f5now + f6now - f4now  - f7now - f8now)/ro;
    v_y = (f1now + f5now + f8now - f3now  - f6now - f7now)/ro;
    v_y -= density[i] * 0.0005f;

    float speedcap = 0.22f;
    if (v_x < -speedcap) v_x = -speedcap;
    if (v_x >  speedcap) v_x =  speedcap;
    if (v_y < -speedcap) v_y = -speedcap;
    if (v_y >  speedcap) v_y =  speedcap;

    velocity[i*2] = v_x;
    velocity[i*2 +1] = v_y;

    float uu = v_x * v_x;
    float vv = v_y * v_y;
    float uv = v_x * v_y;
    
    
    // Calculate equilibrium f's

    f0eq = ro * faceq1 * (1.0f - 1.5f * (uu + vv));
    f1eq = ro * faceq2 * (1.0f + 3.0f * v_y + 3.f * vv - 1.5f * uu);
    f2eq = ro * faceq2 * (1.0f + 3.0f * v_x + 3.f * uu - 1.5f * vv);
    f3eq = ro * faceq2 * (1.0f - 3.0f * v_y + 3.f * vv - 1.5f * uu);
    f4eq = ro * faceq2 * (1.0f - 3.0f * v_x + 3.f * uu - 1.5f * vv);
    f5eq = ro * faceq3 * (1.0f + 3.0f * v_x + 3.f * v_y + 3.f * uu + 3.f * vv + 9.f * uv);
    f6eq = ro * faceq3 * (1.0f + 3.0f * v_x - 3.f * v_y + 3.f * uu + 3.f * vv - 9.f * uv);
    f7eq = ro * faceq3 * (1.0f - 3.0f * v_x - 3.f * v_y + 3.f * uu + 3.f * vv + 9.f * uv);
    f8eq = ro * faceq3 * (1.0f - 3.0f * v_x + 3.f * v_y + 3.f * uu + 3.f * vv - 9.f * uv);

    // Do collisions
    f0_data[i] += (f0eq - f0_data[i]) / tau;
    f1_data[i] += (f1eq - f1_data[i]) / tau;
    f2_data[i] += (f2eq - f2_data[i]) / tau;
    f3_data[i] += (f3eq - f3_data[i]) / tau;
    f4_data[i] += (f4eq - f4_data[i]) / tau;
    f5_data[i] += (f5eq - f5_data[i]) / tau;
    f6_data[i] += (f6eq - f6_data[i]) / tau;
    f7_data[i] += (f7eq - f7_data[i]) / tau;
    f8_data[i] += (f8eq - f8_data[i]) / tau;
}

extern "C" void initializeSolverData(int width, int height)
{
    const int size = width * height * sizeof(float);
    cutilSafeCall( cudaMalloc((void **)&density_data, size) );
    cutilSafeCall( cudaMalloc((void **)&injection_data, size * 2) );
    cutilSafeCall( cudaMalloc((void **)&velocity_data, size * 2) );
    cutilSafeCall( cudaMalloc((void **)&f0_data, size) );
    cutilSafeCall( cudaMalloc((void **)&f1_data, size) );
    cutilSafeCall( cudaMalloc((void **)&f2_data, size) );
    cutilSafeCall( cudaMalloc((void **)&f3_data, size) );
    cutilSafeCall( cudaMalloc((void **)&f4_data, size) );
    cutilSafeCall( cudaMalloc((void **)&f5_data, size) );
    cutilSafeCall( cudaMalloc((void **)&f6_data, size) );
    cutilSafeCall( cudaMalloc((void **)&f7_data, size) );
    cutilSafeCall( cudaMalloc((void **)&f8_data, size) );
    
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cutilSafeCall( cudaMallocArray( &density_array, &channelDesc, width, height ));
    cutilSafeCall( cudaMallocArray( &f1_array, &channelDesc, width, height )); 
    cutilSafeCall( cudaMallocArray( &f2_array, &channelDesc, width, height )); 
    cutilSafeCall( cudaMallocArray( &f3_array, &channelDesc, width, height )); 
    cutilSafeCall( cudaMallocArray( &f4_array, &channelDesc, width, height )); 
    cutilSafeCall( cudaMallocArray( &f5_array, &channelDesc, width, height )); 
    cutilSafeCall( cudaMallocArray( &f6_array, &channelDesc, width, height )); 
    cutilSafeCall( cudaMallocArray( &f7_array, &channelDesc, width, height )); 
    cutilSafeCall( cudaMallocArray( &f8_array, &channelDesc, width, height )); 
    
    dim3 grid = dim3(width/TILE_DIM, height/TILE_DIM);
    dim3 block = dim3(TILE_DIM, TILE_DIM);
    reset_kernel<<<grid, block>>>( f0_data, f1_data, f2_data, f3_data, 
                                f4_data, f5_data, f6_data, f7_data, f8_data, 
                                density_data, injection_data, velocity_data, width);
}

extern "C" 
void destroySolverData()
{
    printf("reset");
    cutilSafeCall( cudaFree(density_data) );
    cutilSafeCall( cudaFree(injection_data) );
    cutilSafeCall( cudaFree(velocity_data) );
    cutilSafeCall( cudaFree(f0_data) );
    cutilSafeCall( cudaFree(f1_data) );
    cutilSafeCall( cudaFree(f2_data) );
    cutilSafeCall( cudaFree(f3_data) );
    cutilSafeCall( cudaFree(f4_data) );
    cutilSafeCall( cudaFree(f5_data) );
    cutilSafeCall( cudaFree(f6_data) );
    cutilSafeCall( cudaFree(f7_data) );
    cutilSafeCall( cudaFree(f8_data) );

    cutilSafeCall(cudaFreeArray(f1_array));
    cutilSafeCall(cudaFreeArray(f2_array));
    cutilSafeCall(cudaFreeArray(f3_array));
    cutilSafeCall(cudaFreeArray(f4_array));
    cutilSafeCall(cudaFreeArray(f5_array));
    cutilSafeCall(cudaFreeArray(f6_array));
    cutilSafeCall(cudaFreeArray(f7_array));
    cutilSafeCall(cudaFreeArray(f8_array));
    cudaDeviceReset();
}

extern "C" 
void getDisplayField(int width, int height, unsigned char * obstable, unsigned char *outImage)
{
    const int obstacleLength = width * height * sizeof(unsigned char);
    
    unsigned char *d_obstacle;
    cutilSafeCall( cudaMalloc((void **)&d_obstacle, obstacleLength) );
    cutilSafeCall( cudaMemcpy(d_obstacle, obstable, obstacleLength, cudaMemcpyHostToDevice) );   
    
    const int size = width * height * sizeof(unsigned char) * 3;
	
    unsigned char *d_Out;
    cutilSafeCall( cudaMalloc((void **)&d_Out, size) );
    
    dim3 grid = dim3(width/TILE_DIM, height/TILE_DIM);
    dim3 block = dim3(TILE_DIM, TILE_DIM);
    
    show_scalar<<< grid, block>>>(density_data, d_obstacle, d_Out, width, height);

    cutilSafeCall( cudaMemcpy( outImage, d_Out,  size, cudaMemcpyDeviceToHost));
    
    cutilSafeCall( cudaFree(d_Out)); 
    cutilSafeCall( cudaFree(d_obstacle)); 
}

extern "C" 
void advanceSolver(int width, int height, float *impulse, unsigned char * obstable)
{
    const int obstacleLength = width * height * sizeof(unsigned char);
    
    unsigned char *d_obstacle;
    cutilSafeCall( cudaMalloc((void **)&d_obstacle, obstacleLength) );
    cutilSafeCall( cudaMemcpy(d_obstacle, obstable, obstacleLength, cudaMemcpyHostToDevice) );   
    
    
    const int size = width * height * sizeof(float);
    cutilSafeCall( cudaMemcpy(injection_data, impulse, size * 2, cudaMemcpyHostToDevice) );    
    
    dim3 grid = dim3(width/TILE_DIM, height/TILE_DIM);
    dim3 block = dim3(TILE_DIM, TILE_DIM);
    
    inject_energy<<<grid, block>>>(density_data, 
                                   f1_data, f2_data, f3_data, f4_data,
                                   f5_data, f6_data, f7_data, f8_data, 
                                   injection_data, d_obstacle, width, height);
    
    cutilSafeCall( cudaMemcpy(impulse, injection_data, size * 2, cudaMemcpyDeviceToHost));
    
    boundary_condition_kernel<<<grid, block>>>(f1_data, f2_data, f3_data, f4_data,
                                   f5_data, f6_data, f7_data, f8_data, 
                                   d_obstacle, width);
      
    

    collide_kernel<<<grid, block>>>( f0_data, f1_data, f2_data, f3_data, 
                                f4_data, f5_data, f6_data, f7_data, f8_data, 
                                velocity_data, density_data,
                                width);

    cutilSafeCall( cudaMemcpyToArray( f1_array, 0, 0, f1_data, size, cudaMemcpyDeviceToDevice));
    cutilSafeCall( cudaMemcpyToArray( f2_array, 0, 0, f2_data, size, cudaMemcpyDeviceToDevice));
    cutilSafeCall( cudaMemcpyToArray( f3_array, 0, 0, f3_data, size, cudaMemcpyDeviceToDevice));
    cutilSafeCall( cudaMemcpyToArray( f4_array, 0, 0, f4_data, size, cudaMemcpyDeviceToDevice));
    cutilSafeCall( cudaMemcpyToArray( f5_array, 0, 0, f5_data, size, cudaMemcpyDeviceToDevice));
    cutilSafeCall( cudaMemcpyToArray( f6_array, 0, 0, f6_data, size, cudaMemcpyDeviceToDevice));
    cutilSafeCall( cudaMemcpyToArray( f7_array, 0, 0, f7_data, size, cudaMemcpyDeviceToDevice));
    cutilSafeCall( cudaMemcpyToArray( f8_array, 0, 0, f8_data, size, cudaMemcpyDeviceToDevice));

    f1_tex.filterMode = cudaFilterModePoint;
    f1_tex.addressMode[0] = cudaAddressModeClamp;
    f1_tex.addressMode[1] = cudaAddressModeClamp;
    f1_tex.normalized = false;
    CUDA_SAFE_CALL(cudaBindTextureToArray(f1_tex, f1_array));

    f2_tex.filterMode = cudaFilterModePoint;
    f2_tex.addressMode[0] = cudaAddressModeClamp;
    f2_tex.addressMode[1] = cudaAddressModeClamp;
    f2_tex.normalized = false;
    CUDA_SAFE_CALL(cudaBindTextureToArray(f2_tex, f2_array));

    f3_tex.filterMode = cudaFilterModePoint;
    f3_tex.addressMode[0] = cudaAddressModeClamp;
    f3_tex.addressMode[1] = cudaAddressModeClamp;
    f3_tex.normalized = false;
    CUDA_SAFE_CALL(cudaBindTextureToArray(f3_tex, f3_array));

    f4_tex.filterMode = cudaFilterModePoint;
    f4_tex.addressMode[0] = cudaAddressModeClamp;
    f4_tex.addressMode[1] = cudaAddressModeClamp;
    f4_tex.normalized = false;
    CUDA_SAFE_CALL(cudaBindTextureToArray(f4_tex, f4_array));

    f5_tex.filterMode = cudaFilterModePoint;
    f5_tex.addressMode[0] = cudaAddressModeClamp;
    f5_tex.addressMode[1] = cudaAddressModeClamp;
    f5_tex.normalized = false;
    CUDA_SAFE_CALL(cudaBindTextureToArray(f5_tex, f5_array));

    f6_tex.filterMode = cudaFilterModePoint;
    f6_tex.addressMode[0] = cudaAddressModeClamp;
    f6_tex.addressMode[1] = cudaAddressModeClamp;
    f6_tex.normalized = false;
    CUDA_SAFE_CALL(cudaBindTextureToArray(f6_tex, f6_array));

    f7_tex.filterMode = cudaFilterModePoint;
    f7_tex.addressMode[0] = cudaAddressModeClamp;
    f7_tex.addressMode[1] = cudaAddressModeClamp;
    f7_tex.normalized = false;
    CUDA_SAFE_CALL(cudaBindTextureToArray(f7_tex, f7_array));

    f8_tex.filterMode = cudaFilterModePoint;
    f8_tex.addressMode[0] = cudaAddressModeClamp;
    f8_tex.addressMode[1] = cudaAddressModeClamp;
    f8_tex.normalized = false;
    CUDA_SAFE_CALL(cudaBindTextureToArray(f8_tex, f8_array));

    stream_kernel<<<grid, block>>>(f1_data, f2_data, f3_data, f4_data,
                                   f5_data, f6_data, f7_data, f8_data, 
                                   width);

    CUDA_SAFE_CALL(cudaUnbindTexture(f1_tex));
    CUDA_SAFE_CALL(cudaUnbindTexture(f2_tex));
    CUDA_SAFE_CALL(cudaUnbindTexture(f3_tex));
    CUDA_SAFE_CALL(cudaUnbindTexture(f4_tex));
    CUDA_SAFE_CALL(cudaUnbindTexture(f5_tex));
    CUDA_SAFE_CALL(cudaUnbindTexture(f6_tex));
    CUDA_SAFE_CALL(cudaUnbindTexture(f7_tex));
    CUDA_SAFE_CALL(cudaUnbindTexture(f8_tex));

    cutilSafeCall( cudaMemcpyToArray( density_array, 0, 0, density_data, size, cudaMemcpyDeviceToDevice));

    densityTex.addressMode[0] = cudaAddressModeClamp;
    densityTex.addressMode[1] = cudaAddressModeClamp;
    densityTex.filterMode = cudaFilterModeLinear;
    densityTex.normalized = false;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cutilSafeCall( cudaBindTextureToArray(densityTex, density_array, channelDesc));
    advect_density<<<grid, block>>>(density_data, velocity_data, width, height);
    cutilSafeCall(cudaUnbindTexture(densityTex));
    
    cutilSafeCall( cudaFree(d_obstacle));
}

#endif 