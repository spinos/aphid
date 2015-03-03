#include "narrowphase_implement.h"
#include <bvh_math.cu>

__global__ void writeObjectPointToCache_kernel(float3 * dstPos,
        float3 * dstVel,
        float3 * srcPos,
        float3 * srcVel,
        uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

	if(ind >= maxInd) return;
	dstPos[ind] = srcPos[ind];
	dstVel[ind] = srcVel[ind];
}

__global__ void writeObjectIndexToCache_kernel(uint4 * dstInd,
    uint4 * srcInd,
    uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

	if(ind >= maxInd) return;
	dstInd[ind] = srcInd[ind];
}

__global__ void computeSeparateAxis_kernel(float3 * dst, 
    uint2 * pairs,
    float3 * pos, float3 * vel, 
    uint4* tetrahedron, 
    uint * pointStart, uint * indexStart,
    uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

	if(ind >= maxInd) return;
	
	uint objA = extractObjectInd(pairs[ind].x);
	uint objB = extractObjectInd(pairs[ind].y);
	uint elmA = extractElementInd(pairs[ind].x);
	uint elmB = extractElementInd(pairs[ind].y);
	
	uint indA = pointStart[objA] + tetrahedron[indexStart[objA] + elmA].z;
	uint indB = pointStart[objB] + tetrahedron[indexStart[objB] + elmB].z;
	
	float3 pa = float3_add(pos[indA], scale_float3_by(vel[indA], 0.01667) );
	float3 pb = float3_add(pos[indB], scale_float3_by(vel[indB], 0.01667));
	dst[ind] = scale_float3_by(float3_add(pa, pb), 0.5);
}

extern "C" {
void narrowphaseWriteObjectToCache(float3 * dstPos,
        float3 * dstVel,
        uint4 * dstInd,
        float3 * srcPos,
        float3 * srcVel,
        uint4 * srcInd,
        uint numPoints,
		uint numTetradedrons)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(numPoints, 512);
    dim3 grid(nblk, 1, 1);
    
    writeObjectPointToCache_kernel<<< grid, block >>>(dstPos, dstVel, srcPos, srcVel, numPoints);
    
    nblk = iDivUp(numTetradedrons, 512);
    grid.x = nblk;
    
    writeObjectIndexToCache_kernel<<< grid, block >>>(dstInd, srcInd, numTetradedrons);
}

void narrowphaseComputeSeparateAxis(float3 * dst,
		uint2 * pairs,
		float3 * pos,
		float3 * vel,
		uint4 * ind,
		uint * pointStart, uint * indexStart,
		uint numOverlappingPairs)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(numOverlappingPairs, 512);
    dim3 grid(nblk, 1, 1);
    
    computeSeparateAxis_kernel<<< grid, block >>>(dst, pairs, pos, vel, ind, pointStart, indexStart, numOverlappingPairs);
}

}
