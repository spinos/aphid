#include "lambert_implement.h"

__global__ void 
lambert_kernel(float3* pos, unsigned int width, unsigned int height, float reflectance)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    float phi = 3.1415927f * 2.f / (float) width * x;
    float theta = 3.1415927f * 0.5f / (float) height * y;
    float r =  reflectance / 3.14159265;

    pos[y*width+x] = make_float3(r * sin(theta) * cos(phi), r * sin(theta) * sin(phi), r * cos(theta));
}

extern "C" void lambert(float3 *pos, unsigned width, unsigned height, float reflectance)
{
    dim3 block(8, 8, 1);
    dim3 grid(width / block.x, height / block.y, 1);
    lambert_kernel<<< grid, block>>>(pos, width, height, reflectance);
}
