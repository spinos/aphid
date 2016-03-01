#include "cu/ImageBase.cuh"

__global__ void showTile_kernel(uint * pix, 
                                float * depth)
{
    uint tileInd = blockIdx.y * gridDim.x + blockIdx.x;
    
    uint ind = tileInd * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    
    int r = 255 * (float)blockIdx.x / (float)gridDim.x;
    int g = 255 * (float)blockIdx.y / (float)gridDim.y;
    
	pix[ind] = encodeRGB(r, g, 0); 
	depth[ind] = 1e20f;
}
