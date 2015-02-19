#include "broadphase_implement.h"

__global__ void resetPairCounts_kernel(uint * dst, uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

	if(ind >= maxInd) return;

	dst[ind] = 0;
}

extern "C" {

void broadphaseResetPairCounts(uint * dst, uint num)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(num, 512);
    dim3 grid(nblk, 1, 1);
    resetPairCounts_kernel<<< grid, block >>>(dst, num);
}

}

