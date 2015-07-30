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

namespace trianglesys {
void formTetrahedronAabbs(Aabb * dst,
                                float3 * pos,
                                float3 * vel,
                                float timeStep,
                                uint4 * tets,
                                uint numTriangles)
{
    int tpb = CALC_TETRA_AABB_NUM_THREADS;

    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(numTriangles<<2, tpb);
    
    dim3 grid(nblk, 1, 1);
    formTriangleAabbs_kernel<<< grid, block >>>(dst, pos, vel, timeStep, tets, numTriangles<<2);
}
}
