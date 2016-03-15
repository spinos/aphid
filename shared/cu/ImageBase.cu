#include "ImageBase.cuh"

namespace imagebase {

__global__ void resetImage_kernel(uint * pix, 
                                float * depth,
								uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	pix[ind] = 0; 
	depth[ind] = 0.f;
}

__global__ void resetImage2_kernel(uint * pix, 
                                float * nearDepth,
								float * farDepth,
								uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	pix[ind] = 0; 
	nearDepth[ind] = 0.f;
	farDepth[ind] = 1e28f;
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

void resetImage(uint * pix,
            float * nearDepth,
			float * farDepth,
            int blockx,
            uint n)
{
    dim3 block(blockx, 1, 1);
    int nblk = getNumBlock(n, blockx);
    dim3 grid(nblk, 1, 1);
    
    resetImage2_kernel<<< grid, block >>>(pix, 
        nearDepth,
		farDepth,
        n);
}

}
