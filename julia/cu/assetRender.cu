#include "assetRender.cuh"

namespace assr {

void setRenderRect(int * src)
{ cudaMemcpyToSymbol(c_renderRect, src, 16); }

void setFrustum(float * src)
{ cudaMemcpyToSymbol(c_frustumVec, src, 72); }

void drawPyramid(uint * color,
                float * depth,
                int blockx,
                int gridx, int gridy,
				void * planes,
				void * bounding)
{
    dim3 block(blockx, blockx, 1);
    dim3 grid(gridx, gridy, 1);
    
    assetPyrmaid_kernel<<< grid, block >>>(color, 
        depth,
        (float4 *)planes,
        (Aabb *)bounding);
}

}
