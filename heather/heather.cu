#include "heather_implement.h"
namespace CUU {
    
__global__ void fillImage_kernel( ushort4 * dstCol,  float * dstDep,  ushort4 * srcCol,  float * srcDep, uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

    unsigned pix;
    int i;
    for(i=0; i < 4; i++) {
        pix = ind * 4 + i;
        if(pix >= maxInd) return;
        
        dstCol[pix] = srcCol[pix];
        dstDep[pix] = srcDep[pix];
	}
}

__global__ void mixImage_kernel( ushort4 * dstCol,  float * dstDep,  ushort4 * srcCol,  float * srcDep, uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

    unsigned pix;
    float d;
    int i;
    for(i=0; i < 4; i++) {
        pix = ind * 4 + i;
        if(pix >= maxInd) return;
	
	    d = srcDep[pix];
        if(d < 0.1) return;
        if(d < dstDep[pix] || dstDep[pix] < 0.1f) {
            dstCol[pix] = srcCol[pix];
            dstDep[pix] = srcDep[pix];
        }
    }
}

extern "C" {
void heatherFillImage( ushort4 * dstCol,  float * dstDep,  ushort4 * srcCol,  float * srcDep, uint numPix)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(numPix/4, 512);
    dim3 grid(nblk, 1, 1);
    fillImage_kernel<<< grid, block >>>(dstCol, dstDep, srcCol, srcDep, numPix);
}

void heatherMixImage( ushort4 * dstCol,  float * dstDep,  ushort4 * srcCol,  float * srcDep, uint numPix)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(numPix/4, 512);
    dim3 grid(nblk, 1, 1);
    mixImage_kernel<<< grid, block >>>(dstCol, dstDep, srcCol, srcDep, numPix);
}
}
}
