#include "WorldRender.cuh"

namespace wldr {

void setRenderRect(int * src)
{ cudaMemcpyToSymbol(c_renderRect, src, 16); }

void setFrustum(float * src)
{ cudaMemcpyToSymbol(c_frustumVec, src, 72); }

void render(uint * pix,
            float * nearDepth,
			float * farDepth,
			void * ropes,
            int blockx,
            int gridx, int gridy)
{
    dim3 block(blockx, blockx, 1);
    dim3 grid(gridx, gridy, 1);
    
    twoCube_kernel<<< grid, block >>>(pix, 
        nearDepth,
		farDepth,
		(Rope *)ropes);
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
