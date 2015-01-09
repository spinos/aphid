#include <assert.h>
#include <cutil_inline.h>
#include "plane_implement.h"

__global__ void 
hemisphere_kernel(float4* pos, unsigned dim, unsigned maxInd, float gridSize, float alpha)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

	if(ind >= maxInd) return;

	unsigned gv = ind / (dim + 1);
	unsigned gu = ind - gv * (dim + 1);
	
	float4 sum;
	sum.x = gridSize * gu - gridSize * dim / 2;
	sum.y = -10.0 + 2.5 * gridSize * sin(alpha + 3.14 * 2.0 * gu / dim) + .75 *  gridSize * cos(alpha * 1.25 - 3.14 * 3.0 * gv / dim);
	sum.z = gridSize * gv - gridSize * dim / 2;
	sum.w = 1.f;
	
	pos[ind] = sum;
}

extern "C" void wavePlane(float4 *pos, unsigned numGrids, float gridSize, float alpha)
{
	dim3 block(512, 1, 1);
	
	unsigned np = (numGrids + 1) * (numGrids + 1);
    unsigned nblk = iDivUp(np, 512);
    
    dim3 grid(nblk, 1, 1);
	hemisphere_kernel<<< grid, block >>>(pos, numGrids, np, gridSize, alpha);
}
