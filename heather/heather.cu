#include "heather_implement.h"
//https://devtalk.nvidia.com/default/topic/370791/cuda-programming-and-performance/what-about-half-float-/
// extern __device__ unsigned short __float2half_rn(float);
// extern __device__ float __half2float(unsigned short);
namespace CUU {
    
__global__ void fillImage_kernel( ushort4 * dstCol,  float * dstDep,  ushort4 * srcCol,  float * srcDep, 
                                 uint numPix, uint pixWidth, uint imageWidth,
                                 uint reduceRatio)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x*bw + tx;
    int y = blockIdx.y*bh + ty;
    
    int ind = y * pixWidth + x;
    if(ind >= numPix) return;

    int loc = y * reduceRatio * imageWidth + x * reduceRatio;
    dstCol[ind] = srcCol[loc];
    dstDep[ind] = srcDep[loc];
}

__global__ void mixImage_kernel( ushort4 * dstCol, float * dstDep,  ushort4 * srcCol,  float * srcDep,
                                uint numPix, uint pixWidth, uint imageWidth,
                      uint reduceRatio)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bw = blockDim.x;
    int bh = blockDim.y;
    int x = blockIdx.x*bw + tx;
    int y = blockIdx.y*bh + ty;
    
    int ind = y * pixWidth + x;
    if(ind >= numPix) return;

    int loc = y * reduceRatio * imageWidth + x * reduceRatio;
    float d = srcDep[loc];
    if(d < 0.1) return;
    if(d < dstDep[ind] || dstDep[ind] < 0.1f) {
         dstCol[ind] = srcCol[loc];
         dstDep[ind] = srcDep[loc];
    }
}

extern "C" {
void heatherFillImage( ushort4 * dstCol, float * dstDep,  ushort4 * srcCol,  float * srcDep, 
                      uint imageWidth, uint imageHeight,
                      uint reduceRatio)
{
    uint pixWidth = imageWidth / reduceRatio;
    uint pixHeight = imageHeight / reduceRatio;
    dim3 block(16, 16, 1);
    dim3 grid(iDivUp(pixWidth, 16), iDivUp(pixHeight, 16), 1);
    uint numPix = pixWidth * pixHeight;
    fillImage_kernel<<< grid, block >>>(dstCol, dstDep, srcCol, srcDep, numPix, pixWidth, imageWidth, reduceRatio);
    cudaDeviceSynchronize();
}

void heatherMixImage( ushort4 * dstCol, float * dstDep,  ushort4 * srcCol,  float * srcDep, 
                     uint imageWidth, uint imageHeight,
                      uint reduceRatio)
{
    uint pixWidth = imageWidth / reduceRatio;
    uint pixHeight = imageHeight / reduceRatio;
    dim3 block(16, 16, 1);
    dim3 grid(iDivUp(pixWidth, 16), iDivUp(pixHeight, 16), 1);
    uint numPix = pixWidth * pixHeight;
    mixImage_kernel<<< grid, block >>>(dstCol, dstDep, srcCol, srcDep, numPix, pixWidth, imageWidth, reduceRatio);
    cudaDeviceSynchronize();
}
}
}
