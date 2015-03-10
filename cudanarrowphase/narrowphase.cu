#include "narrowphase_implement.h"
#include <bvh_math.cu>
#include <gjk_math.cu>
#include <CudaBase.h>

#define GJK_BLOCK_SIZE 64

inline __device__ float totalSpeed(float3 * vA, float3 * vB)
{
    return (float3_length2(vA[0]) + float3_length2(vA[1]) +
            float3_length2(vA[2]) + float3_length2(vA[2]) +
            float3_length2(vB[0]) + float3_length2(vB[1]) +
            float3_length2(vB[2]) + float3_length2(vB[2]));
}

inline __device__ void extractTetrahedron(MovingTetrahedron & tet, uint start, const uint4 & vertices, float3 * pos, float3 * vel)
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

inline __device__ float maxProjectSpeedAlong(const float3 * v, const float3 & d)
{
    float r = float3_dot(d, v[0]);
    float r1 = float3_dot(d, v[1]);
    if(r1 > r) r = r1;
    r1 = float3_dot(d, v[2]);
    if(r1 > r) r = r1;
    r1 = float3_dot(d, v[3]);
    if(r1 > r) r = r1;
    return r;
}

__global__ void computeSeparateAxis_kernel(ContactData * dstContact,
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
	BarycentricCoordinate coord;
	
	computeSeparateDistance(sS[threadIdx.x], sPrxA[threadIdx.x], sPrxB[threadIdx.x], GJK_THIN_MARGIN, ctc, dstContact[ind].separateAxis, 
	    coord);
	
	interpolatePointAB(sS[threadIdx.x], coord, dstContact[ind].localA, dstContact[ind].localB);
}

__global__ void computeTimeOfImpact_kernel(ContactData * dstContact,
    uint2 * pairs,
    float3 * pos, float3 * vel, 
    uint4 * tetrahedron, 
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
	
	if(totalSpeed(tA.v, tB.v) < 1e-8) {
	    dstContact[ind].timeOfImpact = 1e8;
	    return;
	}
	
	progressTetrahedron(sPrxA[threadIdx.x], tA, 0.f);
	progressTetrahedron(sPrxB[threadIdx.x], tB, 0.f);

	ClosestPointTestContext ctc;
	BarycentricCoordinate coord;
	float4 sas;
	computeSeparateDistance(sS[threadIdx.x], sPrxA[threadIdx.x], sPrxB[threadIdx.x], 0.f, ctc, sas, 
	    coord);
// intersected	
	if(sas.w < 1.f) {
	    dstContact[ind].timeOfImpact = 1e8;
	    return;
	}
	
	float separateDistance = float4_length(sas);
// within thin shell margin
	if(separateDistance < GJK_THIN_MARGIN2) {
	    dstContact[ind].timeOfImpact = 0.f;
	    dstContact[ind].separateAxis = sas;
        interpolatePointAB(sS[threadIdx.x], coord, dstContact[ind].localA, dstContact[ind].localB);
        return;
	}

// use thin shell margin
	separateDistance -= GJK_THIN_MARGIN2;
	
	float3 nor;
	float closeInSpeed;
	float toi = 0.f;
	int i = 0;
    while (i<GJK_MAX_NUM_ITERATIONS) {
        nor = float3_normalize(float3_from_float4(sas));
        closeInSpeed = maxProjectSpeedAlong(tB.v, nor)
                        - maxProjectSpeedAlong(tA.v, nor);
// going apart       
        if(closeInSpeed < 1e-8) { 
            dstContact[ind].timeOfImpact = 1e8;
            
// for debug purpose
            dstContact[ind].separateAxis = sas;
            interpolatePointAB(sS[threadIdx.x], coord, dstContact[ind].localA, dstContact[ind].localB);
        
            break;
        }
        
        toi += separateDistance / closeInSpeed * .732f;
// too far away       
        if(toi > GJK_STEPSIZE) { 
            dstContact[ind].timeOfImpact = toi;
            
// for debug purpose
            dstContact[ind].separateAxis = sas;
            interpolatePointAB(sS[threadIdx.x], coord, dstContact[ind].localA, dstContact[ind].localB);
        
            break;   
        }
        
        progressTetrahedron(sPrxA[threadIdx.x], tA, toi);
        progressTetrahedron(sPrxB[threadIdx.x], tB, toi);
        
        computeSeparateDistance(sS[threadIdx.x], sPrxA[threadIdx.x], sPrxB[threadIdx.x], GJK_THIN_MARGIN, ctc, sas, 
            coord); 
// use last step       
        if(sas.w < 1.f) { 
            break;
        }
// output
        dstContact[ind].timeOfImpact = toi;
        dstContact[ind].separateAxis = sas;
        interpolatePointAB(sS[threadIdx.x], coord, dstContact[ind].localA, dstContact[ind].localB);
        
        separateDistance = float4_length(sas);
// close enough
        if(separateDistance < .001f) { 
            break;
        }
        
        i++;
    }
}

__global__ void computeValidPairs_kernel(uint* dstCounts, 
                    ContactData * srcContact, 
                    uint numContacts, 
                    uint scanBufferLength)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;

	if(ind >= scanBufferLength) return;
	
	dstCounts[ind] = 0;
	if(ind >= numContacts) {
	    return;
	}

	dstCounts[ind] = (srcContact[ind].timeOfImpact < GJK_STEPSIZE);	
}

__global__ void squeezeContactPairs_kernel(uint2 * dstPairs, uint2 * srcPairs,
                                    ContactData * dstContact, ContactData *srcContact,
									uint * counts, uint * packLocs, 
									uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	
	if(!counts[ind]) return;
	
	const uint toLoc = packLocs[ind];
	dstPairs[toLoc] = srcPairs[ind];
	dstContact[toLoc] = srcContact[ind];
}

extern "C" {

void narrowphaseComputeSeparateAxis(ContactData * dstContact,
		uint2 * pairs,
		float3 * pos,
		float3 * vel,
		uint4 * ind,
		uint * pointStart, uint * indexStart,
		uint numOverlappingPairs)
{
    dim3 block(GJK_BLOCK_SIZE, 1, 1);
    unsigned nblk = iDivUp(numOverlappingPairs, GJK_BLOCK_SIZE);
    dim3 grid(nblk, 1, 1);
    
    computeSeparateAxis_kernel<<< grid, block >>>(dstContact, pairs, pos, vel, ind, pointStart, indexStart, numOverlappingPairs);
}

void narrowphaseComputeTimeOfImpact(ContactData * dstContact,
		uint2 * pairs,
		float3 * pos,
		float3 * vel,
		uint4 * ind,
		uint * pointStart, uint * indexStart, 
		uint numOverlappingPairs)
{   
    dim3 block(GJK_BLOCK_SIZE, 1, 1);
    unsigned nblk = iDivUp(numOverlappingPairs, GJK_BLOCK_SIZE);
    dim3 grid(nblk, 1, 1);
    
    computeTimeOfImpact_kernel<<< grid, block >>>(dstContact, pairs, pos, vel, ind, pointStart, indexStart, numOverlappingPairs);
}

void narrowphaseComputeValidPairs(uint * dstCounts, 
        ContactData * srcContact,
        uint numContacts, 
        uint scanBufferLength)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(scanBufferLength, 512);
    dim3 grid(nblk, 1, 1);
    
    computeValidPairs_kernel<<< grid, block >>>(dstCounts, srcContact, numContacts, scanBufferLength);
}

void narrowphaseSqueezeContactPairs(uint2 * dstPairs, uint2 * srcPairs,
                                    ContactData * dstContact, ContactData *srcContact,
									uint * counts, uint * packLocs, 
									uint maxInd)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(maxInd, 512);
    dim3 grid(nblk, 1, 1);
    
    squeezeContactPairs_kernel<<< grid, block >>>(dstPairs, srcPairs,
                                    dstContact, srcContact,
									counts, packLocs, 
									maxInd);
}

}
