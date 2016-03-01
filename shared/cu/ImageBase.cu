#include "ImageBase.cuh"

namespace imagebase {

__global__ void resetImage_kernel(uint * pix, 
                                float * depth,
								uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	pix[ind] = 0; 
	depth[ind] = 1e20f;
}

void resetImage(uint * pix,
            float * depth,
            int blockx,
            uint n)
{
    dim3 block(blockx, 1, 1);
    int nblk = getNumBlock(n, blockx);
    dim3 grid(nblk, 1, 1);
    
    resetImage_kernel<<< grid, block >>>(pix, 
        depth,
        n);
}

}
