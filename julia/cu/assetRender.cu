#include "assetRender.cuh"

namespace assr {

void setRenderRect(int * src)
{ cudaMemcpyToSymbol(c_renderRect, src, 16); }

void setFrustum(float * src)
{ cudaMemcpyToSymbol(c_frustumVec, src, 72); }

void drawCube(uint * color,
                float * nearDepth,
                float * farDepth,
				int blockx,
                int gridx, int gridy,
                void * branches,
				void * leaves,
				void * ropes,
				int * indirections,
				void * primitives
                )
{
    dim3 block(blockx, blockx, 1);
    dim3 grid(gridx, gridy, 1);
    
    if(blockx == 8) {
        assetBox_kernel<64> <<< grid, block, 16000 >>>(color, 
            nearDepth,
            farDepth,
            (NTreeBranch4 *)branches,
            (NTreeLeaf *)leaves,
            (Rope *)ropes,
            indirections,
            (Voxel *)primitives);
	}
	else if(blockx == 16) {
	    assetBox_kernel<256> <<< grid, block, 16000 >>>(color, 
            nearDepth,
            farDepth,
            (NTreeBranch4 *)branches,
            (NTreeLeaf *)leaves,
            (Rope *)ropes,
            indirections,
            (Voxel *)primitives);
	}
}

}
