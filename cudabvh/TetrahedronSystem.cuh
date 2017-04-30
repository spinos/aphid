#ifndef TETRAHEDRONSYSTEM_CUH
#define TETRAHEDRONSYSTEM_CUH

#include "bvh_math.cuh"
#include "Aabb.cuh"
#define CALC_TETRA_AABB_NUM_THREADS 512

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

__global__ void formTetrahedronAabbs_kernel(Aabb *dst, float3 * pos, float3 * vel, float h, 
                                                            uint4 * tetrahedronVertices, 
                                                            unsigned maxNumPerTetVs)
{
    __shared__ float3 sP0[CALC_TETRA_AABB_NUM_THREADS];
    __shared__ float3 sP1[CALC_TETRA_AABB_NUM_THREADS];
    
    uint idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx >= maxNumPerTetVs) return;
	
	uint itet = idx>>2;
	uint ivert = idx & 3;
	uint * vtet = & tetrahedronVertices[itet].x;
	
	uint iv = vtet[ivert];
	
	sP0[threadIdx.x] = pos[iv];
	sP1[threadIdx.x] = float3_progress(pos[iv], vel[iv], h);
	__syncthreads();
	
	if(ivert > 0) return;
	
	Aabb res;
	resetAabb(res);
	
	expandAabb(res, sP0[threadIdx.x]);
	expandAabb(res, sP1[threadIdx.x]);
	expandAabb(res, sP0[threadIdx.x + 1]);
	expandAabb(res, sP1[threadIdx.x + 1]);
	expandAabb(res, sP0[threadIdx.x + 2]);
	expandAabb(res, sP1[threadIdx.x + 2]);
	expandAabb(res, sP0[threadIdx.x + 3]);
	expandAabb(res, sP1[threadIdx.x + 3]);
	
	dst[itet] = res;
}

__global__ void formTetrahedronAabbsImpulsed_kernel(Aabb *dst, 
                                                    float3 * pos, 
                                                    float3 * vel, 
                                                    float3 * deltaVel,
                                                    float h, 
                                                            uint4 * tetrahedronVertices, 
                                                            unsigned maxNumPerTetVs)
{
    __shared__ float3 sP0[CALC_TETRA_AABB_NUM_THREADS];
    __shared__ float3 sP1[CALC_TETRA_AABB_NUM_THREADS];
    
    uint idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx >= maxNumPerTetVs) return;
	
	uint itet = idx>>2;
	uint ivert = idx & 3;
	uint * vtet = & tetrahedronVertices[itet].x;
	
	uint iv = vtet[ivert];
	
    float3 impulsedVel = vel[iv];
    float3_add_inplace(impulsedVel, deltaVel[iv]);
	sP0[threadIdx.x] = pos[iv];
	sP1[threadIdx.x] = float3_progress(pos[iv], impulsedVel, h);
	__syncthreads();
	
	if(ivert > 0) return;
	
	Aabb res;
	resetAabb(res);
	
	expandAabb(res, sP0[threadIdx.x]);
	expandAabb(res, sP1[threadIdx.x]);
	expandAabb(res, sP0[threadIdx.x + 1]);
	expandAabb(res, sP1[threadIdx.x + 1]);
	expandAabb(res, sP0[threadIdx.x + 2]);
	expandAabb(res, sP1[threadIdx.x + 2]);
	expandAabb(res, sP0[threadIdx.x + 3]);
	expandAabb(res, sP1[threadIdx.x + 3]);
	
	dst[itet] = res;
}

#endif        //  #ifndef TETRAHEDRONSYSTEM_CUH

