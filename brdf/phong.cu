#include "phong_implement.h"

#include "brdf_common.h"

__global__ void 
phong_kernel(float3* pos, unsigned int width, unsigned int height, float3 V, float3 N, float exposure, int divideByNdotL)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    float3 L = calculateL(width, height, x, y);
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
