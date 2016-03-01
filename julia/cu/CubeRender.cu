#include "CubeRender.cuh"

namespace cuber {

void render(uint * pix,
            float * depth,
            int blockx,
            int gridx, int gridy)
{
    dim3 block(blockx, blockx, 1);
    dim3 grid(gridx, gridy, 1);
    
    showTile_kernel<<< grid, block >>>(pix, 
        depth);
}

}
