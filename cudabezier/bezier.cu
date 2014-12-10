#include "bezier_implement.h"

__global__ void 
hemisphere_kernel(float4* pos, float3 * cvs, unsigned width)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	unsigned seg = (y*width+x) / 100;

	float wei = (float)((y*width+x)) / 100.f - seg;
	
	float4 sum;
	sum.x = cvs[seg].x * (1.f - wei) + cvs[seg + 1].x * wei;
	sum.y = cvs[seg].y * (1.f - wei) + cvs[seg + 1].y * wei;
	sum.z = cvs[seg].z * (1.f - wei) + cvs[seg + 1].z * wei;
	sum.w = 1.f;
	
	pos[y*width+x] = sum;
}

extern "C" void hemisphere(float4 *pos, float3 * cvs, unsigned numCvs)
{
    dim3 block(16, 16, 1);
    unsigned width = 128;
    unsigned height = (numCvs - 1) * 100 / width;
    dim3 grid(width, height, 1);
    hemisphere_kernel<<< grid, block >>>(pos, cvs, width);
}
