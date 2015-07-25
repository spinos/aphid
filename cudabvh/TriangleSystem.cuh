#ifndef TRIANGLESYSTEM_CUH
#define TRIANGLESYSTEM_CUH

#include "bvh_common.h"
#include "bvh_math.cuh"
#include "Aabb.cuh"

#define CALC_TETRA_AABB_NUM_THREADS 512

__global__ void formTriangleAabbs_kernel(Aabb *dst, float3 * pos, float3 * vel, float h, 
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
	
	dst[itet] = res;
}

__global__ void integrate_kernel(float3 * pos, 
								float3 * vel,
                                float3 * vela,
								float dt, 
								uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
    float3 anchoredVel = vela[ind];
    vel[ind] = anchoredVel;
	float3_add_inplace(pos[ind], scale_float3_by(anchoredVel, dt));
}

#endif        //  #ifndef TRIANGLESYSTEM_CUH

