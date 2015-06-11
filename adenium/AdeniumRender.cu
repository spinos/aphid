#include "AdeniumRender.cuh"

namespace adetrace {
void resetImage(float4 * pix, 
            uint n)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(n, 512);
    dim3 grid(nblk, 1, 1);
    
    resetImage_kernel<<< grid, block >>>(pix,
        n);
}

void setModelViewMatrix(float * src, uint size) 
{
    cudaMemcpyToSymbol(c_modelViewMatrix, src, size);
}

void renderImageOrthographic(float4 * pix,
                uint imageW,
                uint imageH,
                float fovWidth,
                float aspectRatio,
                int2 * nodes,
				Aabb * nodeAabbs)
{
    uint nthread = 8;
    uint nblockX = iDivUp(imageW, nthread);
    uint nblockY = iDivUp(imageH, nthread);
    dim3 block(nthread, nthread, 1);
    dim3 grid(nblockX, nblockY, 1);
    renderImageOrthographic_kernel<<< grid, block >>>(pix,
                        imageW, imageH,
                        fovWidth,
                        aspectRatio,
                        nodes,
                        nodeAabbs);
}

}
