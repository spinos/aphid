#include <assert.h>
#include <cutil_inline.h>
#include "plane_implement.h"

static uint iDivUp(uint dividend, uint divisor){
    return ( (dividend % divisor) == 0 ) ? (dividend / divisor) : (dividend / divisor + 1);
}

__global__ void 
hemisphere_kernel(float4* pos, unsigned width, unsigned dim, unsigned maxInd, float gridSize, float alpha)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	unsigned ind = y*width+x;
	
	if(ind >= maxInd) return;

	unsigned gv = ind / (dim + 1);
	unsigned gu = ind - gv * (dim + 1);
	
	float4 sum;
	sum.x = gridSize * gu - gridSize * dim / 2;
	sum.y = -10.0 + 1.5 * gridSize * sin(alpha + 3.14 * 2.0 * gu / dim) + .5 *  gridSize * cos(alpha * 1.25 - 3.14 * 3.0 * gv / dim);
	sum.z = gridSize * gv - gridSize * dim / 2;
	sum.w = 1.f;
	
	pos[ind] = sum;
}

extern "C" void wavePlane(float4 *pos, unsigned numGrids, float gridSize, float alpha)
{
	dim3 block(8, 8, 1);
	
	unsigned np = (numGrids + 1) * (numGrids + 1);
    unsigned nblk = iDivUp(np, 64);
    unsigned width = nblk * 8;
    
    dim3 grid(nblk, 1, 1);
	hemisphere_kernel<<< grid, block >>>(pos, width, numGrids, np, gridSize, alpha);
}
