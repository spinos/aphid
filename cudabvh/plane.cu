#include <assert.h>
#include <cutil_inline.h>
#include "plane_implement.h"

__global__ void 
hemisphere_kernel(float3* pos, unsigned dim, unsigned maxInd, float gridSize, float alpha, float amplitude)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

	if(ind >= maxInd) return;

	unsigned gv = ind / (dim + 1);
	unsigned gu = ind - gv * (dim + 1);
	
	float3 sum;
	sum.x = gridSize * gu - gridSize * dim / 2;
	sum.y = -10.0 + amplitude * (8.0 * gridSize * sin(alpha * 0.25 + 3.14 * 1.9 * gu / dim) + 4.0 *  gridSize * cos(alpha * .65 - 3.14 * 3.0 * gv / dim));
	sum.z = gridSize * gv - gridSize * dim / 2 + amplitude * sin(alpha * 0.4 );
	
	pos[ind] = sum;
}

extern "C" void wavePlane(float3 *pos, unsigned numGrids, float gridSize, float alpha)
{
	dim3 block(512, 1, 1);
	
	unsigned np = (numGrids + 1) * (numGrids + 1);
    unsigned nblk = iDivUp(np, 512);
    
    dim3 grid(nblk, 1, 1);
	hemisphere_kernel<<< grid, block >>>(pos, numGrids, np, gridSize, alpha, 0.33);
}
