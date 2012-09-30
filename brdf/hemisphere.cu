#include "hemisphere_implement.h"

__global__ void 
hemisphere_kernel(float3* pos, unsigned int width, unsigned int height)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    float phi = 3.1415927f * 2.f / (float) width * x;
    float theta = 3.1415927f * 0.5f / (float) height * y;

    pos[y*width+x] = make_float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
}

extern "C" void hemisphere(float3 *pos, unsigned width, unsigned height)
{
    dim3 block(8, 8, 1);
    dim3 grid(width / block.x, height / block.y, 1);
    hemisphere_kernel<<< grid, block>>>(pos, width, height);
}
