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

void setImageSize(int * src) 
{
    cudaMemcpyToSymbol(c_imageSize, src, 8);
}

void setCameraProp(float * src) 
{
    cudaMemcpyToSymbol(c_cameraProp, src, 8);
}

void renderImage(float4 * pix,
                uint imageW,
                uint imageH,
                int2 * nodes,
				Aabb * nodeAabbs,
				KeyValuePair * elementHash,
				int4 * elementVertices,
				float3 * elementPoints,
				int isOrthographic)
{
    uint nthread = 8;
    uint nblockX = iDivUp(imageW, nthread);
    uint nblockY = iDivUp(imageH, nthread);
    dim3 block(nthread, nthread, 1);
    dim3 grid(nblockX, nblockY, 1);
    if(isOrthographic) {
        OrthographicEye eye;
        renderImage_kernel<64, OrthographicEye> <<< grid, block, 16320 >>>(pix,
                        nodes,
                        nodeAabbs,
				elementHash,
				elementVertices,
				elementPoints,
				eye);
	}
	else {
	    PerspectiveEye eye;
	    renderImage_kernel<64, PerspectiveEye> <<< grid, block, 16320 >>>(pix,
                        nodes,
                        nodeAabbs,
				elementHash,
				elementVertices,
				elementPoints,
				eye);
	}
}

}
