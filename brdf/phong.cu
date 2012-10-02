#include "phong_implement.h"

__device__ 
inline float dot(float3 v0, float3 v1)
{
    return v0.x*v1.x + v0.y*v1.y + v0.z*v1.z;
}

__device__ 
inline float3 scale(float3 v0, float scalar)
{
    float3 res = v0;
    res.x *= scalar;
    res.y *= scalar;
    res.z *= scalar;
    return res;
}

__device__ 
inline float3 minus(float3 v0, float3 v1)
{
    float3 res = v0;
    res.x -= v1.x;
    res.y -= v1.y;
    res.z -= v1.z;
    return res;
}

__device__ 
inline float3 reflect(float3 I, float3 N)
{
    return  minus(scale(N, 2.f * dot(I,N)), I);
}

__global__ void 
phong_kernel(float3* pos, unsigned int width, unsigned int height, float3 V, float3 N, float exposure, int divideByNdotL)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    float phi = 3.1415927f * 2.f / (float) width * x;
    float theta = 3.1415927f * 0.5f / (float) height * y;

    float3 L = make_float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
    float3 R = reflect(L, N);
    
    float val = pow(max(0.f, dot(R,V)), exposure);
    
    if(divideByNdotL > 0)
        val = val / max(10e-3, dot(N,L));

    pos[y*width+x] = scale(L, val);
}

extern "C" void phong_brdf(float3 *pos, unsigned width, unsigned height, float3 V, float3 N, float exposure, int divideByNdotL)
{
    dim3 block(8, 8, 1);
    dim3 grid(width / block.x, height / block.y, 1);
    phong_kernel<<< grid, block>>>(pos, width, height, V, N, exposure, divideByNdotL);
}
