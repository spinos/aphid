#include "simple_implement.h"

__global__ void fillImage_kernel(float4 * dstPix, 
        uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

	int i;
	for(i=0; i < 4; i++) {
	    if(ind * 4 + i >= maxInd) return;
	
	    float r = (float)threadIdx.x/(float)blockDim.x;
	    dstPix[ind * 4 + i] = make_float4(r, 1.f - r, 0.f, 1.f);
	}
}

extern "C" {
void fillImage(float4 * dstPix, 
        uint numPix)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(numPix/4, 512);
    dim3 grid(nblk, 1, 1);
    
    fillImage_kernel<<< grid, block >>>(dstPix, numPix);
}
}
