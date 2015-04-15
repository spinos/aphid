#include "cuFemTetrahedron_implement.h"
#include "bvh_math.cu"
#include "matrix_math.cu"

__global__ void resetRe_kernel(mat33* dst, 
                    uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

	if(ind >= maxInd) return;
	
	set_mat33_identity(dst[ind]);
}

extern "C" {
void cuFemTetrahedron_resetRe(mat33 * d, uint maxInd)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(maxInd, 512);
    dim3 grid(nblk, 1, 1);
    
    resetRe_kernel<<< grid, block >>>(d, maxInd);
}

}
