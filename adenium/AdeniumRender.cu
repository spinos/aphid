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

}
