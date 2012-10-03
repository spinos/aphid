#include "ward_implement.h"

#include "brdf_common.h"

__global__ void 
ward_kernel(float3* pos, unsigned int width, unsigned int height, float3 V, float3 N, float3 X, float3 Y, float alpha_x, float alpha_y, bool anisotropic)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    float3 L = calculateL(width, height, x, y);
    float3 H = normalize(add(L, V));
    
    float ax = alpha_x;
    float ay = anisotropic ? alpha_y : alpha_x;
    float exponent = -2.f * (sqr(dot(H,X) / ax) + sqr(dot(H,Y) / ay)) / (1.f + dot(H, N));
    
    float spec = 1.f / (4.f * 3.1415926f * ax * ay * sqrt(dot(L,N) * dot(V, N)));
    spec *= exp(exponent);
    
    pos[y*width+x] = scale(L, spec);
}

extern "C" void ward_brdf(float3 *pos, unsigned width, unsigned height, float3 V, float3 N, float3 X, float3 Y, float alpha_x, float alpha_y, bool anisotropic)
{
    dim3 block(8, 8, 1);
    dim3 grid(width / block.x, height / block.y, 1);
    ward_kernel<<< grid, block>>>(pos, width, height, V, N, X, Y, alpha_x, alpha_y, anisotropic);
}
