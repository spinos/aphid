#include "narrowphase_implement.h"
#include <bvh_math.cu>
#include <gjk_math.cu>
#include <CudaBase.h>

#define GJK_BLOCK_SIZE 64

inline __device__ void extractTetrahedron(MovingTetrahedron & tet, uint start, uint4 vertices, float3 * pos, float3 * vel)
{
    uint ind = start + vertices.x;
    tet.p[0] = pos[ind];
    tet.v[0] = vel[ind];
    ind = start + vertices.y;
    tet.p[1] = pos[ind];
    tet.v[1] = vel[ind];
    ind = start + vertices.z;
    tet.p[2] = pos[ind];
    tet.v[2] = vel[ind];
    ind = start + vertices.w;
    tet.p[3] = pos[ind];
    tet.v[3] = vel[ind];
}

inline __device__ void progressTetrahedron(TetrahedronProxy & prx, const MovingTetrahedron & tet, float h)
{
    prx.p[0] = float3_add(tet.p[0], scale_float3_by(tet.v[0], h));
    prx.p[1] = float3_add(tet.p[1], scale_float3_by(tet.v[1], h));
    prx.p[2] = float3_add(tet.p[2], scale_float3_by(tet.v[2], h));
    prx.p[3] = float3_add(tet.p[3], scale_float3_by(tet.v[3], h));
}

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

__global__ void computeSeparateAxis_kernel(float4 * dstSA,
        float3 * dstPA, float3 * dstPB, 
        BarycentricCoordinate * dstCoord,                                
    uint2 * pairs,
    float3 * pos, float3 * vel, 
    uint4* tetrahedron, 
    uint * pointStart, uint * indexStart,
    uint maxInd)
{
    __shared__ Simplex sS[GJK_BLOCK_SIZE];
    __shared__ TetrahedronProxy sPrxA[GJK_BLOCK_SIZE];
	__shared__ TetrahedronProxy sPrxB[GJK_BLOCK_SIZE];
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

	if(ind >= maxInd) return;
	
	uint objA = extractObjectInd(pairs[ind].x);
	uint objB = extractObjectInd(pairs[ind].y);
	uint elmA = extractElementInd(pairs[ind].x);
	uint elmB = extractElementInd(pairs[ind].y);
	
	MovingTetrahedron tA;
	MovingTetrahedron tB;
	
	extractTetrahedron(tA, pointStart[objA], tetrahedron[indexStart[objA] + elmA], pos, vel);
	extractTetrahedron(tB, pointStart[objB], tetrahedron[indexStart[objB] + elmB], pos, vel);
	
	progressTetrahedron(sPrxA[threadIdx.x], tA, 0.01667f);
	progressTetrahedron(sPrxB[threadIdx.x], tB, 0.01667f);

	ClosestPointTestContext ctc;
	ctc.referencePoint = make_float3(0.0f, 0.0f, 0.0f);
	computeSeparateDistance(sS[threadIdx.x], sPrxA[threadIdx.x], sPrxB[threadIdx.x], ctc, dstSA[ind], dstCoord[ind]);
	
	interpolatePointAB(sS[threadIdx.x], dstCoord[ind], dstPA[ind], dstPB[ind]);
}

extern "C" {

void narrowphaseComputeSeparateAxis(float4 * dstSA,
        float3 * dstPA, float3 * dstPB,
        BarycentricCoordinate * dstCoord,
		uint2 * pairs,
		float3 * pos,
		float3 * vel,
		uint4 * ind,
		uint * pointStart, uint * indexStart,
		uint numOverlappingPairs)
{
    int tpb = GJK_BLOCK_SIZE;//CudaBase::LimitNThreadPerBlock(60, 48);
    dim3 block(tpb, 1, 1);
    unsigned nblk = iDivUp(numOverlappingPairs, tpb);
    dim3 grid(nblk, 1, 1);
    
    computeSeparateAxis_kernel<<< grid, block >>>(dstSA, dstPA, dstPB, dstCoord, pairs, pos, vel, ind, pointStart, indexStart, numOverlappingPairs);
}

}
