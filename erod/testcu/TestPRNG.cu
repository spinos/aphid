#include <cuPRNG.cuh>
#include "bvh_common.h"
namespace tprng {

__global__ void rand_kernel(float * dst,
                uint * dsd,
                uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	uint sd = ind;
	dst[ind] = HybridTaus(sd);
	dsd[ind] = sd;
}

void rand(float * dst,
            unsigned * sd,
                unsigned maxInd)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(maxInd, 512);
    dim3 grid(nblk, 1, 1);
    
    rand_kernel<<< grid, block >>>(dst,
        sd,
        maxInd);
}

}
