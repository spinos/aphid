#include "hemisphere_implement.h"
#include "brdf_common.h"

__global__ void 
hemisphere_kernel(float3* pos, unsigned width)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    float3 pin = pos[y*width+x];

    pos[y*width+x] = normalize(pin);
}

extern "C" void hemisphere(float3 *pos, unsigned numVertices)
{
    dim3 block(8, 8, 1);
    unsigned width = 128;
    unsigned height = numVertices / width;
    dim3 grid(width / block.x, height / block.y, 1);
    hemisphere_kernel<<< grid, block>>>(pos, width);
}
