#ifndef TETRAHEDRONSYSTEM_CUH
#define TETRAHEDRONSYSTEM_CUH

#include "bvh_math.cuh"

__global__ void tetrahedronSystemIntegrate_kernel(float3 * o_position, float3 * i_velocity, 
                                    float dt, uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

	if(ind >= maxInd) return;
	
	o_position[ind] = float3_progress(o_position[ind], i_velocity[ind], dt);
}

template<int VicinityLength>
__global__ void writeVicinity_kernel(int * vicinities,
                    int * indices,
                    int * offsets,
                    uint maxInd)
{
    uint ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	const int start = VicinityLength * ind;
	const int minInd = offsets[ind];
	int curInd = offsets[ind+1] - 1;
	int i = 0;
	for(;i<VicinityLength;i++) {
	    if(curInd < minInd) 
	        vicinities[start + i] = -1;
	    else 
	        vicinities[start + i] = indices[curInd--];
	}
}
#endif        //  #ifndef TETRAHEDRONSYSTEM_CUH

