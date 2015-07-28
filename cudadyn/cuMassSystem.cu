#include <bvh_common.h>

__global__ void computeMass_kernel(float * dst,
                float * mass0,
                uint * anchored,
                float scale,
                uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	float m0 = mass0[ind];
	if(anchored[ind] > 0) dst[ind] = m0 * scale;
}

namespace masssystem {
void computeMass(float * dst,
                float * mass0,
                uint * anchored,
                float scale,
                uint maxInd)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(maxInd, 512);
    dim3 grid(nblk, 1, 1);
    
    computeMass_kernel<<< grid, block >>>(dst,
        mass0,
        anchored,
        scale,
        maxInd);
}
}
