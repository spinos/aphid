#include "cooktorrance_implement.h"

#include "brdf_common.h"

__global__ void 
cooktorrance_kernel(float3* pos, unsigned int width, float3 V, float3 N, float m, float f0)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    float3 L = calculateL(pos, width, x, y);
    float3 H = normalize(add(L, V));
    float NdotH = dot(N, H);
    float VdotH = dot(V, H);
    float NdotL = dot(N, L);
    float NdotV = dot(N, V);

    float D = beckmann(m, NdotH);
    float F = fresnel(f0, VdotH);

    NdotH = NdotH + NdotH;
    
    float G = min(2.f * NdotH * NdotL / VdotH, 2.f * NdotH * NdotV / VdotH);
    G = min(1.f, G);
    
    float val = D * G * F / NdotL / NdotV / 3.1415927f;
    
    pos[y*width+x] = scale(L, val);
}

extern "C" void cooktorrance_brdf(float3 *pos, unsigned numVertices, unsigned width, float3 V, float3 N, float m, float f0)
{
    dim3 block(8, 8, 1);
    unsigned height = numVertices / width;
    dim3 grid(width / block.x, height / block.y, 1);
    cooktorrance_kernel<<< grid, block>>>(pos, width, V, N, m, f0);
}
