#include "tetrahedronSystem_implement.h"
#include "bvh_math.cu"

__global__ void tetrahedronSystemIntegrate_kernel(float3 * o_position, float3 * i_velocity, 
                                    float dt, uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

	if(ind >= maxInd) return;
	
	o_position[ind] = float3_progress(o_position[ind], i_velocity[ind], dt);
}

extern "C" {
void tetrahedronSystemIntegrate(float3 * o_position, float3 * i_velocity, 
                                    float dt, uint n)
{
    dim3 block(512, 1, 1);
	unsigned nblk = iDivUp(n, 512);
	dim3 grid(nblk, 1, 1);
	
	tetrahedronSystemIntegrate_kernel<<< grid, block >>>(o_position, i_velocity, dt, n);
}
}

