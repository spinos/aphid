#include "heather_implement.h"
namespace CUU {
    
__global__ void fillImage_kernel( ushort4 * dstCol,  float * dstDep,  ushort4 * srcCol,  float * srcDep, uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

	if(ind >= maxInd) return;
	
	dstCol[ind] = srcCol[ind];
	dstDep[ind] = srcDep[ind];
}

__global__ void mixImage_kernel( ushort4 * dstCol,  float * dstDep,  ushort4 * srcCol,  float * srcDep, uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

	if(ind >= maxInd) return;
	
	float d = srcDep[ind];
	if(d < 0.1) return;
	if(d < dstDep[ind] || dstDep[ind] < 0.1f) {
	    dstCol[ind] = srcCol[ind];
	    dstDep[ind] = srcDep[ind];
	}
}

extern "C" {
void heatherFillImage( ushort4 * dstCol,  float * dstDep,  ushort4 * srcCol,  float * srcDep, uint numPix)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(numPix, 512);
    dim3 grid(nblk, 1, 1);
    fillImage_kernel<<< grid, block >>>(dstCol, dstDep, srcCol, srcDep, numPix);
}

void heatherMixImage( ushort4 * dstCol,  float * dstDep,  ushort4 * srcCol,  float * srcDep, uint numPix)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(numPix, 512);
    dim3 grid(nblk, 1, 1);
    mixImage_kernel<<< grid, block >>>(dstCol, dstDep, srcCol, srcDep, numPix);
}
}
}
