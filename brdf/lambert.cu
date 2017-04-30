#include "lambert_implement.h"
#include "brdf_common.h"

__global__ void 
lambert_kernel(float3* pos, unsigned int width, float reflectance)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    float3 pin = pos[y*width+x];
    float r =  reflectance / 3.14159265;

    pos[y*width+x] = scale(normalize(pin), r);
}

extern "C" void lambert_brdf(float3 *pos, unsigned numVertices, unsigned width, float reflectance)
{
    dim3 block(8, 8, 1);
    unsigned height = numVertices / width;
    dim3 grid(width / block.x, height / block.y, 1);
    lambert_kernel<<< grid, block>>>(pos, width, reflectance);
}
