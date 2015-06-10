#include "bvh_math.cuh"

__global__ void resetImage_kernel(float4 * pix, 
								uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	pix[ind] = make_float4(0.f, 0.f, 0.f, 1e20f);
}

