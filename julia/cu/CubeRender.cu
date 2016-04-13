#include "CubeRender.cuh"

namespace cuber {

void setRenderRect(int * src)
{ cudaMemcpyToSymbol(c_renderRect, src, 16); }

void setFrustum(float * src)
{ cudaMemcpyToSymbol(c_frustumVec, src, 72); }

void render(uint * pix,
            float * depth,
            int blockx,
            int gridx, int gridy)
{
    dim3 block(blockx, blockx, 1);
    dim3 grid(gridx, gridy, 1);
    
    oneCube_kernel<<< grid, block >>>(pix, 
        depth);
}

void drawPyramid(uint * color,
                float * depth,
                int blockx,
                int gridx, int gridy,
				void * planes,
				void * bounding)
{
    dim3 block(blockx, blockx, 1);
    dim3 grid(gridx, gridy, 1);
    
    onePyrmaid_kernel<<< grid, block >>>(color, 
        depth,
        (float4 *)planes,
        (Aabb *)bounding);
}

const float cubefaces[] = {
-1, 0, 0,
 1, 0, 0,
 0,-1, 0,
 0, 1, 0,
 0, 0,-1,
 0, 0, 1
};

void setBoxFaces()
{ cudaMemcpyToSymbol(c_ray_box_face, cubefaces, 72); }

}
